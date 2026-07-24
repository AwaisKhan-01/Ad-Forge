"""
Parses structured text, extracts Explicit Keyframes, and composites them in PARALLEL.
Enforces strict asset isolation, natural language prompts, typography overlays, and clamped denoising.
Located in: app/new_pipeline/frame_designer.py
"""
import os
import re
import uuid
import json
import subprocess
import fal_client
import concurrent.futures
from typing import List, Dict
from app.new_pipeline.uploader import upload_video

IMAGE_EDIT_MODEL = "fal-ai/nano-banana-2/edit"  # kept for backwards-compat / default
TEXT_LLM_MODEL = "fal-ai/llama-3.1-8b-instruct"
FRAMES_DIR = "frames"

# ==========================================
# IMAGE MODEL REGISTRY
# ==========================================
# Each entry describes how to talk to one image-edit model. The UI lets the user
# pick which model + (for models that support it) which size/quality to use.
#   - "endpoint": fal model id
#   - "supports_size": whether the model takes an image_size preset (gpt-image-2)
#                      vs only aspect_ratio (nano-banana)
#   - "sizes": allowed image_size preset strings (empty if not supported)
#   - "qualities": allowed quality tiers (empty if not supported)
#   - "pricing": per-image USD cost keyed by (size, quality); used by CostTracker
IMAGE_MODELS = {
    "nano-banana-2-edit": {
        "label": "Nano Banana 2 (Edit)",
        "endpoint": "fal-ai/nano-banana-2/edit",
        "supports_size": False,
        "sizes": [],
        "qualities": [],
        # Flat estimate — nano-banana has no size/quality tiers on fal.
        "flat_price": 0.015,
    },
    "gpt-image-2": {
        "label": "GPT Image 2 (Edit)",
        "endpoint": "fal-ai/gpt-image-2/edit",
        "supports_size": True,
        "sizes": ["landscape_4_3", "square_hd", "portrait_4_3", "landscape_16_9"],
        "qualities": ["low", "medium", "high"],
        # fal per-image pricing by (image_size, quality). Sizes map to pixel dims:
        #   landscape_4_3 = 1024x768, portrait_4_3 = 768x1024, landscape_16_9 = 1024x576
        #   square_hd = 1024x1024. 16:9 priced same tier as the 1024x768 bracket.
        "pricing": {
            ("landscape_4_3", "low"): 0.005, ("landscape_4_3", "medium"): 0.037, ("landscape_4_3", "high"): 0.145,
            ("portrait_4_3", "low"): 0.005, ("portrait_4_3", "medium"): 0.037, ("portrait_4_3", "high"): 0.145,
            ("landscape_16_9", "low"): 0.005, ("landscape_16_9", "medium"): 0.037, ("landscape_16_9", "high"): 0.145,
            ("square_hd", "low"): 0.006, ("square_hd", "medium"): 0.053, ("square_hd", "high"): 0.211,
        },
    },
}

DEFAULT_IMAGE_MODEL = "nano-banana-2-edit"


def image_price(model_key: str, image_size: str = "", quality: str = "") -> float:
    """Per-image USD cost for the chosen model/size/quality (best-effort estimate)."""
    m = IMAGE_MODELS.get(model_key, IMAGE_MODELS[DEFAULT_IMAGE_MODEL])
    if not m["supports_size"]:
        return m.get("flat_price", 0.015)
    return m.get("pricing", {}).get((image_size, quality), 0.05)

# ==========================================
# TYPOGRAPHY PRESET REGISTRY
# ==========================================
TYPOGRAPHY_PRESETS = {
    "NO_TEXT": "no text, no typography, no letters, no logos, no branding, clean composition",
    "chunky_display": "bold chunky irregular hand-drawn display typeface, interlocking blocky letters, playful raw graphic design layout, thick solid weights",
    "elegant_script_mix": "high-contrast elegant spencerian script typography mixed with clean bold geometric sans-serif accent lettering, high-fashion luxury editorial text layout",
    "streetwear_graffiti": "grungy streetwear stenciled typography text graphic, distressed spray-paint texture, urban stencil art style, sharp aggressive letter angles, clean crisp monochrome edges",
    "brutalist_geometric": "modular avant-garde blocky typography text design, solid uniform stroke weight lettering, minimalist structural brutalist graphic layout, sharp clean corners",
    "retro_3d_editorial": "vintage 1970s editorial script text graphic with dramatic extruded 3D drop shadow, swooping decorative custom lettering swashes, clean retro advertisement composition",
    "psychedelic_bubble": "retro 70s psychedelic fluid bubble typography, curvy soft-serif lettering, playful decorative star sparkles accents, clean smooth vector art text overlay",
    "liquid_cursive": "high-contrast organic liquid typography text overlay, flowing interlocking cursive curves, bold abstract fluid font graphic layout",
    "raw_expressionist": "distorted jagged papercut font style, irregular blocky expressionist typography, raw textured distressed zine-style graphic layout"
}

# Words that mark a string as an audio DESCRIPTION rather than spoken words.
# A legacy combined "Audio:" line holding e.g. "upbeat electronic music" must never
# be treated as dialogue: it would be spoken aloud by the on-screen actor AND
# rendered as on-screen typography.
_AUDIO_DESCRIPTOR_HINTS = (
    "music", "score", "soundtrack", "beat", "melody", "instrumental", "synth",
    "piano", "guitar", "drum", "bass", "ambient", "ambience", "sfx",
    "sound effect", "whoosh", "riser", "swell", "no vocals", "upbeat",
    "background audio", "audio cue", "silence", "silent",
)

def _speech_only(text: str) -> str:
    """Return `text` only if it reads like literal spoken words.

    Returns "" when the string is clearly a music/SFX description, so descriptive
    audio can never leak into the spoken-dialogue path.
    """
    t = (text or "").strip()
    if not t or t in ("...", "-", "..", "n/a", "none", "N/A"):
        return ""
    low = t.lower()
    # Quoted text is almost always real dialogue, even if it mentions music.
    if '"' in t or "'" in t:
        return t
    if any(h in low for h in _AUDIO_DESCRIPTOR_HINTS):
        return ""
    return t

def parse_keyframe_to_seconds(kf_str: str) -> float:
    try:
        clean_str = kf_str.strip().replace(" ", "")
        if ':' in clean_str:
            m, s = clean_str.split(':')
            return int(m) * 60 + float(s)
        return float(clean_str)
    except Exception:
        return 0.0

def extract_frame(video_path: str, timestamp_sec: float) -> str:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
    # FRAME-ACCURATE SEEK: place -ss AFTER -i (output seeking). With -ss BEFORE -i
    # (input seeking), ffmpeg snaps to the nearest prior keyframe (I-frame), which
    # in fast-cut ads can be 1-3s off — so the extracted frame shows a *different*
    # shot than the visual_prompt describes. Output seeking decodes to the exact
    # timestamp. Slightly slower per frame, but negligible for a single still.
    cmd = [
        "ffmpeg", "-i", video_path, "-ss", str(timestamp_sec),
        "-frames:v", "1", "-q:v", "2", out_path, "-y"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out_path if os.path.exists(out_path) else None
    except subprocess.CalledProcessError:
        return None

def _compile_diffusion_prompt(visual_text: str, is_text_banned: bool) -> str:
    ban_instruction = "CRITICAL: Omit any mentions of text, letters, typography, or branding." if is_text_banned else "Preserve descriptions of text or logos exactly as they appear."
    
    compiler_prompt = f"""You are a Stable Diffusion Prompt Engineer. Rewrite this scene description into a clean, cinematic prompt for a STATIC image generation model.

SCENE DESCRIPTION: "{visual_text}"

RULES:
1. FREEZE TIME: You are describing a still photograph. Isolate the primary action and discard the rest. Strip out all sequential actions, timelines, or transitions (e.g., REMOVE phrases like "then falls back", "transitioning to", "pans", "zooms").
2. Strip out conversational AI meta-text (e.g., "The scene shows", "Make sure to morph", "In this frame").
3. DO NOT comma-separate every word. Write in natural, descriptive, grammatical phrases.
4. Focus purely on lighting, subject, and static composition.
5. PRESERVE IMAGE REFERENCE TAGS: Crucially, if the description contains layout anchors like #image1 or #image2, you MUST retain them exactly as they are in your final output.
6. {ban_instruction}

OUTPUT FORMAT: Return ONLY the clean prompt string. No quotes, no intro.
"""
    try:
        result = fal_client.subscribe(
            TEXT_LLM_MODEL, 
            arguments={
                "prompt": compiler_prompt, 
                "system_prompt": "You are a prompt compiler. Output clean, strictly static descriptive phrases while preserving layout reference tags.", 
                "max_tokens": 100, 
                "temperature": 0.1
            }
        )
        return result.get("output", "").strip()
    except Exception:
        clean = visual_text.lower()
        bad_phrases = ["the scene shows", "appears", "transform", "camera pans", "morph human likeness", "replace main subject", "then", "transitioning", "transitions"]
        for b in bad_phrases: 
            clean = clean.replace(b, "")
        return clean.strip()

def parse_extracted_scenes(extracted_text: str) -> List[Dict]:
    scenes = []

    # 1) Quick JSON fallback: if the VLM returned a JSON object, try to parse it.
    #    Use brace-balanced extraction rather than a greedy `.*` regex, which would
    #    over-match past the intended object when other braces follow in the text.
    try:
        candidate = None
        start = extracted_text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(extracted_text)):
                ch = extracted_text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = extracted_text[start:i + 1]
                        break
        if candidate:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "scenes" in parsed:
                for s in parsed["scenes"]:
                    assets = s.get("assigned_assets", [])
                    scenes.append({
                        "scene_number": str(s.get("scene_number", "")),
                        "timestamp": s.get("timestamp", ""),
                        "keyframe_time": s.get("keyframe_time", ""),
                        "visual_prompt": s.get("visual_prompt", ""),
                        "motion": s.get("motion", ""),
                        "spoken_dialogue": s.get("dialogue", "") or s.get("spoken_dialogue", "") or _speech_only(s.get("audio", "")),
                        "sfx_music": s.get("sfx_music", ""),
                        "on_screen_text": s.get("on_screen_text", ""),
                        "transition": s.get("transition", "cut"),
                        "assigned_assets": assets if isinstance(assets, list) else [],
                    })
                return scenes
    except Exception:
        pass

    # 2) Line-by-line block parser.
    #    The previous single mega-regex made every field optional, so all fields
    #    collapsed into the last group — the ENTIRE scene body was swept into
    #    assigned_assets and every digit in the text (timestamps, keyframes) became
    #    a bogus asset index. e.g. "[1]" became [0,0,0,3,0,2,1,1], appending the
    #    user's single asset ~6 times per scene. Instead: split on "SCENE <n>"
    #    headers, then read known "Label: value" lines. Robust to field ordering.
    LABEL_MAP = {
        "timestamp": "timestamp", "timecode": "timestamp",
        "keyframe": "keyframe_time", "keyframe time": "keyframe_time",
        "visual": "visual_prompt", "visual prompt": "visual_prompt",
        "motion": "motion",
        "spoken dialogue": "spoken_dialogue", "spoken": "spoken_dialogue", "dialogue": "spoken_dialogue",
        "on-screen text": "on_screen_text", "on screen text": "on_screen_text", "onscreen text": "on_screen_text",
        "transition": "transition",
        "sfx/music": "sfx_music", "sfx": "sfx_music", "music": "sfx_music",
        "assigned assets": "assigned_assets", "assigned": "assigned_assets", "assets": "assigned_assets",
    }

    # re.split with a capture group yields: [pre, num1, body1, num2, body2, ...]
    blocks = re.split(r"(?im)^\s*SCENE\s+(\d+)\s*$", extracted_text)
    for i in range(1, len(blocks), 2):
        scene_number = blocks[i].strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""

        fields = {
            "scene_number": scene_number, "timestamp": "", "keyframe_time": "",
            "visual_prompt": "", "motion": "", "spoken_dialogue": "",
            "sfx_music": "", "on_screen_text": "", "transition": "cut", "assigned_assets": [],
        }
        raw_audio = ""  # analyzer emits a single "Audio:" line

        for line in body.splitlines():
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            key = LABEL_MAP.get(label.strip().lower())
            value = value.strip()
            if key == "assigned_assets":
                # Parse ONLY the assets value (e.g. "[1]"), never the whole block.
                try:
                    parsed_list = json.loads(value)
                    fields["assigned_assets"] = parsed_list if isinstance(parsed_list, list) else []
                except Exception:
                    nums = re.findall(r"-?\d+", value)
                    fields["assigned_assets"] = [int(n) for n in nums]
            elif key:
                fields[key] = value
            elif label.strip().lower() == "audio":
                raw_audio = value

        # Legacy single "Audio:" line: route it to dialogue ONLY if it reads like
        # spoken words. Music/SFX descriptions go to sfx_music instead, so they are
        # never spoken aloud or rendered as on-screen typography.
        if raw_audio and not fields["spoken_dialogue"] and not fields["sfx_music"]:
            speech = _speech_only(raw_audio)
            if speech:
                fields["spoken_dialogue"] = speech
            else:
                fields["sfx_music"] = raw_audio

        scenes.append(fields)

    return scenes

def _process_single_scene(scene: Dict, asset_urls: List[str], local_video_path: str, aspect_ratio: str, selected_font_style: str = "chunky_display", image_model: str = DEFAULT_IMAGE_MODEL, image_size: str = "", image_quality: str = "high", prompt_override: str = "", image_urls_override: List[str] = None, motion_graphic: bool = False) -> Dict:
    target_time = parse_keyframe_to_seconds(scene.get('keyframe_time', '00:00'))
    print(f"\n[EXTRACTING] Scene {scene.get('scene_number')} -> Exact Keyframe Target: {target_time:.2f}s")
    
    frame_path = extract_frame(local_video_path, target_time)
    if not frame_path:
        scene["generated_image_url"] = None
        return scene
        
    base_frame_url = upload_video(frame_path)
    image_urls = [base_frame_url]
    
    assigned_indices = scene.get("assigned_assets", [])
    raw_visual_text = scene['visual_prompt']
    
    # --- SURGICAL FIX: Only evaluate spoken_dialogue for text triggers ---
    scene_dialogue = scene.get("spoken_dialogue", "").strip()
    
    # Only words that unambiguously ask for RENDERED TEXT. Removed "interface",
    # "logo" and "displaying": an ordinary shot like "a laptop displaying a
    # dashboard interface" is not a request for typography, but used to force it.
    trigger_words = ["text", "typography", "font", "title card", "caption", "subtitle", "lettering"]
    has_explicit_text_request = any(word in raw_visual_text.lower() for word in trigger_words)
    # Motion graphics carry design copy in on_screen_text instead of dialogue.
    on_screen_text = (scene.get("on_screen_text") or "").strip()
    if motion_graphic and on_screen_text:
        scene_dialogue = on_screen_text
    # Music/SFX descriptions must never count as voiceover (they'd become on-image text).
    has_spoken_voiceover = bool(_speech_only(scene_dialogue))
    # The user's typography choice is authoritative: picking "No Text" bans text
    # even for scenes that have dialogue. Previously this was ignored entirely.
    user_wants_no_text = (selected_font_style == "NO_TEXT")
    is_text_banned = user_wants_no_text or not (has_explicit_text_request or has_spoken_voiceover)

    pre_mapped_text = raw_visual_text
    # --- ASSET MAPPING (1-BASED, AUTHORITATIVE) ---
    # Assets are numbered the way the user sees them in the UI: Asset 1 is the
    # FIRST uploaded file. The VLM is instructed to use the same 1-based scheme,
    # so "Asset N" always means asset_urls[N - 1]. No index guessing happens here:
    # a previous revision treated N as a 0-based array index, which silently
    # served the WRONG file whenever more than one asset was uploaded (Asset 1
    # returned the second upload). Numbering is now consistent across the UI
    # label, the VLM prompt, and this lookup.
    #
    # The tags actually written in the prompt text are the source of truth for
    # WHICH assets this scene needs. assigned_assets is merged in as a fallback
    # for when the VLM populates the array but forgets the inline tag.
    tags_in_text = [int(n) for n in re.findall(r"(?i)#asset\s*(\d+)\b", raw_visual_text)]

    normalized_indices = []
    for raw_idx in list(tags_in_text) + list(assigned_indices):
        try:
            raw_idx = int(raw_idx)
        except (TypeError, ValueError):
            print(f"      [WARNING] Non-integer asset index {raw_idx!r}. Skipping.")
            continue
        if raw_idx not in normalized_indices:
            normalized_indices.append(raw_idx)

    for raw_idx in normalized_indices:
        array_idx = raw_idx - 1  # Asset 1 -> asset_urls[0]

        if not (0 <= array_idx < len(asset_urls)):
            print(f"      [WARNING] Scene requested Asset {raw_idx}, but only "
                  f"{len(asset_urls)} asset(s) were uploaded. Skipping.")
            continue

        image_urls.append(asset_urls[array_idx])
        target_image_tag = f"#image{len(image_urls)}"
        print(f"      [ASSET MAP] Asset {raw_idx} -> upload #{raw_idx} -> {target_image_tag}")

        # Rewrite the tag the VLM wrote (case-insensitive, tolerates "#Asset 1").
        pre_mapped_text = re.sub(rf"(?i)#asset\s*{raw_idx}\b", target_image_tag, pre_mapped_text)

    # Safety net: neutralize any orphan #Asset<N> tags the VLM emitted without a
    # matching array entry, so a literal "#Asset1" can NEVER reach the image model.
    pre_mapped_text = re.sub(r"(?i)#asset\s*\d+\b", "#image1", pre_mapped_text)

    sterilized_phrase = _compile_diffusion_prompt(pre_mapped_text, is_text_banned)

    has_assets = len(image_urls) > 1  # >1 means real user assets are attached

    if motion_graphic:
        # STYLE COMES FROM THE REFERENCE, NOT FROM US. Do not inject any aesthetic
        # vocabulary here: words like "flat", "vector" or "graphic shapes" push the
        # model toward cartoon illustration regardless of what the reference ad
        # actually looks like. #image1 is the reference keyframe, so it carries the
        # real art direction (3D, photoreal, dark, glossy, etc.) — we only describe
        # WHAT changes, and let the reference dictate HOW it looks.
        asset_clause = (" Composite the provided brand assets into the layout, matching the "
                        "existing rendering style, lighting and materials."
                        if has_assets else "")
        design_prompt = (
            f"Match the exact visual style, rendering technique, materials, lighting, colour "
            f"palette and level of realism of #image1. {sterilized_phrase}{asset_clause} "
            f"Keep the same art direction as #image1 — do not switch to a different style, "
            f"and do not simplify it into flat illustration or cartoon artwork."
        )
    elif has_assets:
        # The keyframe (#image1) guides composition/lighting, but the ASSET is the
        # true subject. Phrase it so the model builds the described scene around the
        # asset and uses #image1 as a layout/style reference — NOT as a rigid
        # composition to preserve. This avoids the "keep this laptop-screen layout
        # AND put a person here" contradiction when the keyframe happens to show a
        # different shot type than the described subject.
        design_prompt = (
            f"Generate this scene using #image1 as a composition, framing and lighting reference: "
            f"{sterilized_phrase} "
            f"The subject and content must match the description above; adapt the framing "
            f"from #image1 rather than copying its exact contents."
        )
    else:
        design_prompt = (
            f"Keep #image1 almost exactly as-is; only subtly enhance it. "
            f"{sterilized_phrase}"
        )

    if not is_text_banned:
        font_prompt_modifier = TYPOGRAPHY_PRESETS.get(selected_font_style, TYPOGRAPHY_PRESETS["chunky_display"])
        # Only ever render actual SPOKEN words as typography — never a music cue.
        text_content = _speech_only(scene_dialogue) if has_spoken_voiceover else ""
        if text_content:
            design_prompt += f', featuring integrated typography reading exactly "{text_content}" styled in {font_prompt_modifier}'
        else:
            design_prompt += f', styled with integrated {font_prompt_modifier}'
    else:
        # Text is banned — actually SEND the no-text instruction. Previously the
        # NO_TEXT preset was unreachable dead code, so nothing told the model to
        # keep the frame clean and it happily rendered text anyway.
        design_prompt += f", {TYPOGRAPHY_PRESETS['NO_TEXT']}"

    # For motion graphics the look is inherited from the reference keyframe, so add
    # only quality words — no style words that would override that art direction.
    design_prompt += (", high production value, professional composition, crisp and clean"
                      if motion_graphic
                      else ", cinematic lighting, professional composition, clean")

    # --- Build model-specific args ---
    model_cfg = IMAGE_MODELS.get(image_model, IMAGE_MODELS[DEFAULT_IMAGE_MODEL])
    endpoint = model_cfg["endpoint"]

    if model_cfg["supports_size"]:
        # GPT Image 2: uses image_size preset + quality tier (no strength/aspect_ratio).
        size = image_size if image_size in model_cfg["sizes"] else (model_cfg["sizes"][0] if model_cfg["sizes"] else "square_hd")
        quality = image_quality if image_quality in model_cfg["qualities"] else "high"
        args = {
            "prompt": design_prompt,
            "image_urls": image_urls,
            "image_size": size,
            "quality": quality,
        }
        payload_note = f"{model_cfg['label']} | size={size} quality={quality}"
    else:
        # Nano Banana: aspect_ratio + denoising strength (unchanged behavior).
        # LOW strength preserves the keyframe; higher gives room to swap assets in.
        # Motion graphics must REPLACE the photographic keyframe with a designed
        # composition, so it needs far more freedom than a live-action edit where
        # the frame is being preserved.
        edit_strength = 0.85 if motion_graphic else (0.55 if has_assets else 0.25)
        args = {
            "prompt": design_prompt,
            "image_urls": image_urls,
            "aspect_ratio": aspect_ratio,
            "strength": edit_strength,
            "negative_prompt": ("text, letters, words, typography, captions, subtitles, watermark, "
                                if is_text_banned else "") +
                               "ugly, deformed, blurry text, double letters, messy layout, generic sans-serif, standard subtitle boxes, hallucinated faces, overprocessed, deep-fried AI, cluttered, noisy"
        }
        payload_note = f"{model_cfg['label']} | aspect={aspect_ratio} strength={edit_strength}"

    # --- OPTIONAL USER OVERRIDES (additive; no-ops unless the user edited them
    # in the "inspect inputs" panel). Everything above is unchanged. ---
    if prompt_override and prompt_override.strip():
        args["prompt"] = prompt_override.strip()
    if image_urls_override:
        _cleaned = [u.strip() for u in image_urls_override if u and u.strip()]
        if _cleaned:
            args["image_urls"] = _cleaned

    try:
        print(f"      [SCENE {scene.get('scene_number')} PAYLOAD] {payload_note} | Typography: {selected_font_style} | Prompt: {args['prompt']}")
        result = fal_client.subscribe(endpoint, arguments=args)
        if result and "images" in result:
            scene["generated_image_url"] = result["images"][0]["url"]
            scene["image_error"] = None
        else:
            scene["generated_image_url"] = None
            scene["image_error"] = "Model returned no image (often a silent content-policy rejection)."
    except Exception as e:
        print(f"Error on Scene {scene.get('scene_number')}: {e}")
        scene["generated_image_url"] = None
        scene["image_error"] = str(e) or e.__class__.__name__

    scene["visual_prompt_sterilized"] = sterilized_phrase
    # Persist the EXACT payload sent so the UI can show what produced this frame.
    scene["image_endpoint"] = endpoint
    scene["image_args"] = args
    return scene

def generate_storyboard_frames(extracted_text: str, uploaded_assets: List[str], local_video_path: str, aspect_ratio: str = "9:16", selected_font_style: str = "chunky_display", image_model: str = DEFAULT_IMAGE_MODEL, image_size: str = "", image_quality: str = "high", motion_graphic: bool = False) -> List[Dict]:
    scenes = parse_extracted_scenes(extracted_text)
    if not scenes:
        raise ValueError("Could not parse scenes. Check VLM Agent output format.")

    asset_urls = []
    if uploaded_assets:
        for asset in uploaded_assets:
            if asset:
                url = upload_video(asset) if not asset.startswith("http") else asset
                asset_urls.append(url)

    storyboard_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_process_single_scene, scene, asset_urls, local_video_path, aspect_ratio, selected_font_style, image_model, image_size, image_quality, "", None, motion_graphic): scene for scene in scenes}
        for future in concurrent.futures.as_completed(futures):
            storyboard_results.append(future.result())

    storyboard_results.sort(key=lambda x: int(x["scene_number"]))
    return storyboard_results