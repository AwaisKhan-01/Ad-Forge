"""
Parses structured text, extracts Dead-Center Keyframes, and composites them in PARALLEL.
Enforces strict asset isolation to prevent global asset leak poisoning.
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
from app.pipeline.uploader import upload_video

IMAGE_EDIT_MODEL = "fal-ai/nano-banana-2/edit"
TEXT_LLM_MODEL = "fal-ai/llama-3.1-8b-instruct"
FRAMES_DIR = "frames"

def parse_timestamp_to_center_seconds(ts_str: str) -> float:
    try:
        parts = ts_str.replace(" ", "").split('-')
        def time_to_sec(t_str):
            if ':' in t_str:
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        
        start_sec = time_to_sec(parts[0])
        if len(parts) > 1:
            end_sec = time_to_sec(parts[1])
            duration = end_sec - start_sec
            return start_sec + (duration / 2.0)
        return start_sec + 0.5
    except Exception:
        return 0.0

def extract_frame(video_path: str, timestamp_sec: float) -> str:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
    cmd = [
        "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", out_path, "-y"
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return out_path if os.path.exists(out_path) else None

def _compile_diffusion_prompt(visual_text: str, is_text_banned: bool) -> str:
    ban_instruction = "CRITICAL: Omit any tags related to text, typography, or writing." if is_text_banned else "Preserve exact text strings or asset tags like #Asset0 if present."
    
    compiler_prompt = f"""You are a Stable Diffusion Prompt Engineer. Translate this scene into a highly optimized, comma-separated list of visual tags.

SCENE DESCRIPTION: "{visual_text}"

RULES:
1. Strip out ALL conversational words, verbs, and meta-instructions (e.g., "morph human likeness", "replace main subject").
2. Output ONLY visual keywords.
3. {ban_instruction}
4. Ensure UI elements or logos are described flatly.

OUTPUT FORMAT: Return ONLY the comma-separated string.
"""
    try:
        result = fal_client.subscribe(
            TEXT_LLM_MODEL, 
            arguments={
                "prompt": compiler_prompt, 
                "system_prompt": "You are a compiler. Output only comma-separated tags.", 
                "max_tokens": 100, 
                "temperature": 0.1
            }
        )
        return result.get("output", "").strip()
    except Exception:
        clean = visual_text.lower()
        bad = ["shows", "appears", "transform", "camera", "scene", "morph", "likeness", "replace"]
        for b in bad: clean = clean.replace(b, "")
        clean = re.sub(r'[^a-z0-9\s#]', ' ', clean)
        return ", ".join(list(dict.fromkeys([t.strip() for t in clean.split() if len(t.strip()) > 2])))

def parse_extracted_scenes(extracted_text: str) -> List[Dict]:
    scenes = []
    pattern = re.compile(
        r"SCENE\s+(\d+)\s*"
        r"Timestamp:\s*(.*?)\s*"
        r"Visual:\s*(.*?)\s*"
        r"Motion:\s*(.*?)\s*"
        r"Audio:\s*(.*?)\s*"
        r"Assigned Assets:\s*(.*?)(?=\nSCENE|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    
    matches = pattern.finditer(extracted_text)
    for match in matches:
        try:
            assets_list = json.loads(match.group(6).strip())
        except Exception:
            assets_list = []
            
        scenes.append({
            "scene_number": match.group(1).strip(),
            "timestamp": match.group(2).strip(),
            "visual_prompt": match.group(3).strip(),
            "motion": match.group(4).strip(),
            "audio": match.group(5).strip(),
            "assigned_assets": assets_list
        })
    return scenes

def _process_single_scene(scene: Dict, asset_urls: List[str], local_video_path: str, aspect_ratio: str) -> Dict:
    target_time = parse_timestamp_to_center_seconds(scene['timestamp'])
    print(f"\n[EXTRACTING] Scene {scene.get('scene_number')} -> Centered Timecode: {target_time:.2f}s")
    
    frame_path = extract_frame(local_video_path, target_time)
    if not frame_path:
        scene["generated_image_url"] = None
        return scene
        
    base_frame_url = upload_video(frame_path)
    
    # --- SURGICAL FIX: ISOLATE IMAGE PAYLOADS ---
    # Start with ONLY the extracted base keyframe
    image_urls = [base_frame_url]
    
    # Only append the specific assets assigned to THIS scene by the VLM
    assigned_indices = scene.get("assigned_assets", [])
    for idx in assigned_indices:
        if idx < len(asset_urls):
            image_urls.append(asset_urls[idx])
            
    raw_visual_text = scene['visual_prompt']
    trigger_words = ["text", "word", "typography", "font", "interface", "logo", "branding"]
    is_text_banned = not any(word in raw_visual_text.lower() for word in trigger_words)
    
    sterilized_tags = _compile_diffusion_prompt(raw_visual_text, is_text_banned)
    
    # --- SURGICAL FIX: REMOVE HARDCODED BLIND LOOPS ---
    # We construct a clean prompt. We do NOT append generalized "morph human likeness" junk.
    design_prompt = f"{sterilized_tags}, perfect layout integration, highly detailed"
    
    # Clean up the visual tags to reference the clean mapped assets inside the array context
    for local_idx, absolute_idx in enumerate(assigned_indices):
        design_prompt = design_prompt.replace(f"#Asset{absolute_idx}", f"#Image{local_idx + 2}")

    args = {"prompt": design_prompt, "image_urls": image_urls, "aspect_ratio": aspect_ratio}
    
    try:
        print(f"      [SCENE {scene.get('scene_number')} PAYLOAD] Sending {len(image_urls)} target images. Prompt: {design_prompt}")
        result = fal_client.subscribe(IMAGE_EDIT_MODEL, arguments=args)
        scene["generated_image_url"] = result["images"][0]["url"] if result and "images" in result else None
    except Exception as e:
        print(f"Error on Scene {scene['scene_number']}: {e}")
        scene["generated_image_url"] = None 
        
    scene["visual_prompt_sterilized"] = sterilized_tags
    return scene

def generate_storyboard_frames(extracted_text: str, uploaded_assets: List[str], local_video_path: str, aspect_ratio: str = "9:16") -> List[Dict]:
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
        futures = []
        for scene in scenes:
            futures.append(executor.submit(_process_single_scene, scene, asset_urls, local_video_path, aspect_ratio))
            
        for future in concurrent.futures.as_completed(futures):
            storyboard_results.append(future.result())
            
    storyboard_results.sort(key=lambda x: int(x["scene_number"]))
    return storyboard_results