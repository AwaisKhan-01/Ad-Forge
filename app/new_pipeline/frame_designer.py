"""
Parses structured text, extracts Dead-Center Keyframes, and composites them in PARALLEL.
Features the Diffusion Prompt Sterilizer to prevent image model hallucinations.
Located in: app/new_pipeline/frame_designer.py
"""
import os
import re
import uuid
import subprocess
import fal_client
import concurrent.futures
from typing import List, Dict
from app.pipeline.uploader import upload_video
from app.config import NEMOTRON_MODEL

IMAGE_EDIT_MODEL = "fal-ai/nano-banana-2/edit"
FRAMES_DIR = "frames"

# ==========================================
# 1. DEAD-CENTER EXTRACTION MATH
# ==========================================
def parse_timestamp_to_center_seconds(ts_str: str) -> float:
    """
    Calculates the exact middle of a timestamp range to avoid transition blur
    and guarantee we extract the most stable, accurate frame of the scene.
    """
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
            # Extract exactly in the middle of the scene
            return start_sec + (duration / 2.0)
        
        return start_sec + 0.5
    except Exception as e:
        print(f"Timestamp parsing error: {e}")
        return 0.0

def extract_frame(video_path: str, timestamp_sec: float) -> str:
    """Extracts a single frame from the video using FFmpeg."""
    os.makedirs(FRAMES_DIR, exist_ok=True)
    out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
    cmd = [
        "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", out_path, "-y"
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return out_path if os.path.exists(out_path) else None

# ==========================================
# 2. THE DIFFUSION PROMPT STERILIZER
# ==========================================
def _compile_diffusion_prompt(visual_text: str, is_text_banned: bool) -> str:
    """
    Intercepts the Agent's narrative text and translates it into raw, 
    comma-separated diffusion tags to prevent image model hallucination.
    """
    ban_instruction = "CRITICAL: Omit any tags related to text, typography, or writing." if is_text_banned else "Include exact text strings in quotes if specified."
    
    compiler_prompt = f"""You are a Stable Diffusion Prompt Engineer. Your job is to translate a narrative scene description into a highly optimized, comma-separated list of visual tags.

SCENE DESCRIPTION: "{visual_text}"

RULES:
1. Strip out ALL conversational words, verbs, and meta-instructions (e.g., "Transform this", "Modify that", "The camera shows").
2. Output ONLY visual keywords focusing on Subject, Environment, Lighting, and Aesthetic.
3. {ban_instruction}
4. Ensure UI elements are described flatly (e.g., "flat UI design, minimalist interface").

OUTPUT FORMAT:
Return ONLY the comma-separated string. No introduction.
Example: masterpiece, best quality, flat white UI background, minimalist interface, text 'Autonomous Work'
"""
    try:
        result = fal_client.subscribe(
            NEMOTRON_MODEL, 
            arguments={
                "prompt": compiler_prompt, 
                "system_prompt": "You are a compiler. Output only comma-separated tags.", 
                "max_tokens": 200, 
                "temperature": 0.1
            }
        )
        compiled_tags = result.get("output", "").strip()
        print(f"      [STERILIZED PROMPT] -> {compiled_tags}")
        return compiled_tags
    except Exception as e:
        print(f"Prompt Compiler failed: {e}. Falling back to raw text.")
        return visual_text

# ==========================================
# 3. PARSING & ORCHESTRATION
# ==========================================
def parse_extracted_scenes(extracted_text: str) -> List[Dict[str, str]]:
    """Parses the strictly formatted Pydantic output."""
    scenes = []
    pattern = re.compile(
        r"SCENE\s+(\d+)\s*"
        r"Timestamp:\s*(.*?)\s*"
        r"Visual:\s*(.*?)\s*"
        r"Motion:\s*(.*?)\s*"
        r"Audio:\s*(.*?)(?=\nSCENE|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    
    matches = pattern.finditer(extracted_text)
    for match in matches:
        scenes.append({
            "scene_number": match.group(1).strip(),
            "timestamp": match.group(2).strip(),
            "visual_prompt": match.group(3).strip(),
            "motion": match.group(4).strip(),
            "audio": match.group(5).strip()
        })
    return scenes

def _process_single_scene(scene: Dict, asset_urls: List[str], local_video_path: str, aspect_ratio: str) -> Dict:
    # 1. Dead-Center Extraction to guarantee sharp, non-transition frames
    target_time = parse_timestamp_to_center_seconds(scene['timestamp'])
    print(f"\n[EXTRACTING] Scene {scene.get('scene_number')} -> Centered Timecode: {target_time:.2f}s")
    
    frame_path = extract_frame(local_video_path, target_time)
    
    if not frame_path:
        scene["generated_image_url"] = None
        return scene
        
    base_frame_url = upload_video(frame_path)
    image_urls = [base_frame_url]
    
    raw_visual_text = scene['visual_prompt']
    
    # 2. Assess Typography Ban status based on raw agent text
    trigger_words = ["text", "word", "typography", "font", "color", "red", "pink", "blue", "yellow", "white", "black", "interface", "notification"]
    is_text_banned = not any(word in raw_visual_text.lower() for word in trigger_words)
    
    if not asset_urls:
        raw_visual_text = re.sub(r'#Image\d+', '', raw_visual_text, flags=re.IGNORECASE).strip()

    # 3. Sterilize the prompt into Diffusion Tags
    sterilized_tags = _compile_diffusion_prompt(raw_visual_text, is_text_banned)
    
    # 4. Construct the clean image request without aggressive "Bulldozer" overrides
    design_prompt = f"{sterilized_tags}, perfect layout matching #Image1"
    
    if asset_urls:
        for idx, url in enumerate(asset_urls):
            asset_tag = f"#Image{idx + 2}" 
            image_urls.append(url)
            if asset_tag.lower() in raw_visual_text.lower():
                design_prompt += f", replace main subject with {asset_tag}"
            else:
                design_prompt += f", morph human likeness to match {asset_tag}"
    
    args = {"prompt": design_prompt, "image_urls": image_urls, "aspect_ratio": aspect_ratio}
    
    try:
        result = fal_client.subscribe(IMAGE_EDIT_MODEL, arguments=args)
        scene["generated_image_url"] = result["images"][0]["url"] if result and "images" in result else None
    except Exception as e:
        print(f"Error on Scene {scene['scene_number']}: {e}")
        scene["generated_image_url"] = None 
        
    # We pass the sterilized tags BACK into the dictionary so the Video model gets them too!
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