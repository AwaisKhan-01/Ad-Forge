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
    cmd = [
        "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
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

    # 1) Quick JSON fallback: if the VLM returned a JSON object/array, try to parse it.
    try:
        jmatch = re.search(r"\{\s*\"macro_strategy\".*\}\s*", extracted_text, re.DOTALL)
        if not jmatch:
            jmatch = re.search(r"\[\s*\{.*\}\s*\]", extracted_text, re.DOTALL)
        if jmatch:
            try:
                parsed = json.loads(jmatch.group(0))
                # If parsed is a dict with scenes key
                if isinstance(parsed, dict) and "scenes" in parsed:
                    for s in parsed["scenes"]:
                        scenes.append({
                            "scene_number": str(s.get("scene_number", "")),
                            "timestamp": s.get("timestamp", ""),
                            "keyframe_time": s.get("keyframe_time", ""),
                            "visual_prompt": s.get("visual_prompt", ""),
                            "motion": s.get("motion", ""),
                            "spoken_dialogue": s.get("spoken_dialogue", ""),
                            "sfx_music": s.get("sfx_music", ""),
                            "assigned_assets": s.get("assigned_assets", [])
                        })
                    return scenes
            except Exception:
                pass
    except Exception:
        pass

    # 2) Flexible regex to capture various label variants and tolerate spacing/casing differences
    pattern = re.compile(
        r"SCENE\s+(\d+)\s*"                             # Scene number
        r"(?:Timestamp|Timecode)?:?\s*(.*?)\s*"         # Timestamp or Timecode
        r"(?:Keyframe(?: Time)?|Keyframe)?:?\s*(.*?)\s*"# Keyframe or Keyframe Time
        r"(?:Visual|Visual Prompt)?:?\s*(.*?)\s*"       # Visual prompt
        r"(?:Motion)?:?\s*(.*?)\s*"                    # Motion
        r"(?:Spoken Dialogue|Spoken)?:?\s*(.*?)\s*"    # Spoken dialogue
        r"(?:SFX/Music|SFX|Music)?:?\s*(.*?)\s*"       # SFX/Music
        r"(?:Assigned Assets|Assigned|Assets)?:?\s*(.*?)(?=\nSCENE|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    matches = list(pattern.finditer(extracted_text))
    for match in matches:
        raw_assets = match.group(8).strip()
        assets_list = []
        # Try JSON decode first, then fallback to extracting integers
        try:
            assets_list = json.loads(raw_assets)
            if not isinstance(assets_list, list):
                assets_list = []
        except Exception:
            nums = re.findall(r"\d+", raw_assets)
            assets_list = [int(n) for n in nums] if nums else []

        scenes.append({
            "scene_number": match.group(1).strip(),
            "timestamp": match.group(2).strip(),
            "keyframe_time": match.group(3).strip(),
            "visual_prompt": match.group(4).strip(),
            "motion": match.group(5).strip(),
            "spoken_dialogue": match.group(6).strip(),
            "sfx_music": match.group(7).strip(),
            "assigned_assets": assets_list
        })

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
    for local_idx, absolute_idx in enumerate(assigned_indices):
        if absolute_idx >= len(asset_urls) and absolute_idx > 0:
            corrected_idx = absolute_idx - 1
            print(f"      [SYSTEM] Auto-corrected VLM asset index from {absolute_idx} to {corrected_idx}")
        else:
            corrected_idx = absolute_idx
            
        if corrected_idx < len(asset_urls):
            image_urls.append(asset_urls[corrected_idx])
            target_image_tag = f"#image{len(image_urls)}" 
            pre_mapped_text = re.sub(rf"(?i)#asset{absolute_idx}", target_image_tag, pre_mapped_text)
            pre_mapped_text = re.sub(rf"(?i)#asset{corrected_idx}", target_image_tag, pre_mapped_text)
        else:
            print(f"      [WARNING] VLM requested Asset {absolute_idx}, but it does not exist. Skipping.")

    sterilized_phrase = _compile_diffusion_prompt(pre_mapped_text, is_text_banned)
    design_prompt = f"Using #image1 as the base layout: {sterilized_phrase}"

    if not is_text_banned:
        font_prompt_modifier = TYPOGRAPHY_PRESETS.get(selected_font_style, TYPOGRAPHY_PRESETS["chunky_display"])
        # --- SURGICAL FIX: Inject only the spoken dialogue as typography ---
        text_content = scene_dialogue if has_spoken_voiceover else ""
        if text_content:
            design_prompt += f', featuring integrated typography reading exactly "{text_content}" styled in {font_prompt_modifier}'
        else:
            design_prompt += f', styled with integrated {font_prompt_modifier}'

    design_prompt += ", cinematic lighting, professional composition, clean"

    args = {
        "prompt": design_prompt, 
        "image_urls": image_urls, 
        "aspect_ratio": aspect_ratio,
        "strength": 0.40,
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