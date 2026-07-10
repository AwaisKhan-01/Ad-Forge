"""
Parses text, extracts keyframes, and composites them in PARALLEL.
Features Semantic Time Extraction and Strict Asset Binding.
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

IMAGE_EDIT_MODEL = "fal-ai/nano-banana-2/edit"
FRAMES_DIR = "frames"

def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Parses semantic timestamps written by the VLM into an exact extraction second."""
    try:
        parts = ts_str.replace(" ", "").split('-')
        def time_to_sec(t_str):
            if ':' in t_str:
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        start_sec = time_to_sec(parts[0])
        if len(parts) > 1:
            return (start_sec + time_to_sec(parts[1])) / 2.0
        return start_sec
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

def parse_extracted_scenes(extracted_text: str) -> List[Dict[str, str]]:
    """Parses the strict VLM output into dictionaries."""
    scenes = []
    pattern = re.compile(
        r"SCENE\s+(\d+).*?"
        r"Timestamp:\s*(.*?)(?=Visual:|$).*?"
        r"Visual:\s*(.*?)(?=Motion:|$).*?"
        r"Motion:\s*(.*?)(?=(?:Audio|SFX|Audio Focus|Dialogue Focus):|$).*?"
        r"(?:Audio|SFX|Audio Focus|Dialogue Focus):\s*(.*?)(?=---|\Z|SCENE)",
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
    """Processes a single scene using exact semantic timestamps."""
    target_time = parse_timestamp_to_seconds(scene['timestamp'])
    print(f"[EXTRACTING] Scene {scene.get('scene_number')} -> Target Timecode: {target_time:.2f}s")
    
    frame_path = extract_frame(local_video_path, target_time)
    
    if not frame_path:
        scene["generated_image_url"] = None
        return scene
        
    base_frame_url = upload_video(frame_path)
    
    image_urls = [base_frame_url]
    visual_text = scene['visual_prompt']
    
    design_prompt = (
        f"PHOTOREALISTIC COMPOSITION. CRITICAL: DO NOT GENERATE ANY TEXT, WORDS, OR TYPOGRAPHY.\n"
        f"SCENE: {visual_text}\n"
        f"- #Image1 is the base environment.\n"
    )
    
    # --- STRICT ASSET BINDING ---
    assets_injected = False
    if asset_urls:
        for idx, url in enumerate(asset_urls):
            asset_tag = f"#Image{idx + 2}" 
            if asset_tag.lower() in visual_text.lower():
                image_urls.append(url)
                design_prompt += f"- {asset_tag} is the core visual subject. You MUST seamlessly replace the original actor/subject with {asset_tag}.\n"
                assets_injected = True
                
    if not assets_injected:
        design_prompt += "Enhance this scene to be highly photorealistic while maintaining the core base composition. Ignore external tags if none apply."
    # ----------------------------
    
    args = {"prompt": design_prompt, "image_urls": image_urls, "aspect_ratio": aspect_ratio}
    
    try:
        result = fal_client.subscribe(IMAGE_EDIT_MODEL, arguments=args)
        scene["generated_image_url"] = result["images"][0]["url"] if result and "images" in result else None
    except Exception as e:
        print(f"Error on Scene {scene['scene_number']}: {e}")
        scene["generated_image_url"] = None 
        
    return scene

def generate_storyboard_frames(extracted_text: str, uploaded_assets: List[str], local_video_path: str, aspect_ratio: str = "9:16") -> List[Dict]:
    """Pass 3 Orchestrator"""
    scenes = parse_extracted_scenes(extracted_text)
    if not scenes:
        raise ValueError("Could not parse scenes. Check VLM output format.")
        
    asset_urls = []
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