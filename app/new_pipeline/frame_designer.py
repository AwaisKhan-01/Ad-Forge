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

IMAGE_EDIT_MODEL = "fal-ai/nano-banana-2/edit"
TEXT_LLM_MODEL = "fal-ai/llama-3.1-8b-instruct"
FRAMES_DIR = "frames"

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
                        "spoken_dialogue": s.get("spoken_dialogue", "") or s.get("audio", ""),
                        "sfx_music": s.get("sfx_music", ""),
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
            "sfx_music": "", "assigned_assets": [],
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

        # Route a single "Audio:" line to spoken_dialogue so typography/speech works.
        if raw_audio and not fields["spoken_dialogue"] and not fields["sfx_music"]:
            fields["spoken_dialogue"] = raw_audio

        scenes.append(fields)

    return scenes

def _process_single_scene(scene: Dict, asset_urls: List[str], local_video_path: str, aspect_ratio: str, selected_font_style: str = "chunky_display") -> Dict:
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
    
    trigger_words = ["text", "word", "typography", "font", "interface", "logo", "title card", "displaying"]
    has_explicit_text_request = any(word in raw_visual_text.lower() for word in trigger_words)
    has_spoken_voiceover = bool(scene_dialogue and scene_dialogue not in ["", "..."])
    is_text_banned = not (has_explicit_text_request or has_spoken_voiceover)

    pre_mapped_text = raw_visual_text
    # --- ASSET MAPPING ---
    # The VLM is instructed to write #Asset<N> (0-indexed) and list N in
    # assigned_assets. image_urls[0] is always the base keyframe, so the Kth
    # attached asset becomes #image{K+1}. We drive the mapping from the array
    # (the authoritative source) and normalize the text tag to whatever number
    # the VLM actually wrote for that asset, so casing / off-by-one variants of
    # #Asset all resolve to the correct #imageN instead of leaking through.
    for raw_idx in assigned_indices:
        try:
            raw_idx = int(raw_idx)
        except (TypeError, ValueError):
            print(f"      [WARNING] Non-integer asset index {raw_idx!r}. Skipping.")
            continue

        array_idx = raw_idx
        if not (0 <= array_idx < len(asset_urls)):
            # The model is told to 0-index but frequently 1-indexes ("Asset 1" for
            # the first upload). If the index is exactly one past the end, treat it
            # as the 0-indexed equivalent instead of dropping the asset entirely.
            if raw_idx == len(asset_urls) and raw_idx > 0:
                array_idx = raw_idx - 1
                print(f"      [SYSTEM] Interpreting 1-indexed Asset {raw_idx} as asset[{array_idx}].")
            else:
                print(f"      [WARNING] Scene requested Asset {raw_idx}, "
                      f"but only {len(asset_urls)} asset(s) exist. Skipping.")
                continue

        image_urls.append(asset_urls[array_idx])
        target_image_tag = f"#image{len(image_urls)}"

        # Rewrite whichever tag number the VLM actually wrote for this asset —
        # the raw index it listed in the array — case-insensitive and tolerant of
        # an optional space (#Asset 0). This is the number in the prompt text.
        pre_mapped_text = re.sub(rf"(?i)#asset\s*{raw_idx}\b", target_image_tag, pre_mapped_text)

    # Safety net: neutralize any orphan #Asset<N> tags the VLM emitted without a
    # matching array entry, so a literal "#Asset1" can NEVER reach the image model.
    pre_mapped_text = re.sub(r"(?i)#asset\s*\d+\b", "#image1", pre_mapped_text)

    sterilized_phrase = _compile_diffusion_prompt(pre_mapped_text, is_text_banned)

    has_assets = len(image_urls) > 1  # >1 means real user assets are attached
    if has_assets:
        design_prompt = (
            f"Edit #image1, keeping its composition, camera angle and lighting. "
            f"{sterilized_phrase}"
        )
    else:
        design_prompt = (
            f"Keep #image1 almost exactly as-is; only subtly enhance it. "
            f"{sterilized_phrase}"
        )

    if not is_text_banned:
        font_prompt_modifier = TYPOGRAPHY_PRESETS.get(selected_font_style, TYPOGRAPHY_PRESETS["chunky_display"])
        # --- SURGICAL FIX: Inject only the spoken dialogue as typography ---
        text_content = scene_dialogue if has_spoken_voiceover else ""
        if text_content:
            design_prompt += f', featuring integrated typography reading exactly "{text_content}" styled in {font_prompt_modifier}'
        else:
            design_prompt += f', styled with integrated {font_prompt_modifier}'

    design_prompt += ", cinematic lighting, professional composition, clean"

    # LOW strength preserves the keyframe; higher gives room to swap assets in.
    edit_strength = 0.55 if has_assets else 0.25

    args = {
        "prompt": design_prompt,
        "image_urls": image_urls,
        "aspect_ratio": aspect_ratio,
        "strength": edit_strength,
        "negative_prompt": "ugly, deformed, blurry text, double letters, messy layout, generic sans-serif, standard subtitle boxes, hallucinated faces, overprocessed, deep-fried AI, cluttered, noisy"
    }
    
    try:
        print(f"      [SCENE {scene.get('scene_number')} PAYLOAD] Typography: {selected_font_style} | Prompt: {design_prompt}")
        result = fal_client.subscribe(IMAGE_EDIT_MODEL, arguments=args)
        if result and "images" in result:
            scene["generated_image_url"] = result["images"][0]["url"]
        else:
            scene["generated_image_url"] = None
    except Exception as e:
        print(f"Error on Scene {scene.get('scene_number')}: {e}")
        scene["generated_image_url"] = None 
        
    scene["visual_prompt_sterilized"] = sterilized_phrase
    return scene

def generate_storyboard_frames(extracted_text: str, uploaded_assets: List[str], local_video_path: str, aspect_ratio: str = "9:16", selected_font_style: str = "chunky_display") -> List[Dict]:
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
        futures = {executor.submit(_process_single_scene, scene, asset_urls, local_video_path, aspect_ratio, selected_font_style): scene for scene in scenes}
        for future in concurrent.futures.as_completed(futures):
            storyboard_results.append(future.result())
            
    storyboard_results.sort(key=lambda x: int(x["scene_number"]))
    return storyboard_results