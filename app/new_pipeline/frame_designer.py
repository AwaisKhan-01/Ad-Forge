"""
Parses text, extracts keyframes, and composites them in PARALLEL.
Features Smart Asset Binding, Strict Anti-Text Shielding, and Mathematical Time Slicing.
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

def get_video_duration(video_path: str) -> float:
    """Extracts the exact total duration of the source video in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"FFprobe error reading duration (fallback to 15s): {e}")
        return 15.0

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
    """Parses the strict VLM output and injects structural math metadata."""
    scenes = []
    # Indestructible regex catching any variation, including the new 'Dialogue Focus'
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
        
    # Structural Injection: Tell every scene how many total scenes exist for the math calculation
    total = len(scenes)
    for s in scenes:
        s["total_scenes"] = total
        
    return scenes

def _process_single_scene(scene: Dict, asset_urls: List[str], local_video_path: str, aspect_ratio: str) -> Dict:
    """Processes a single scene using mathematically assigned timestamps to guarantee full video coverage."""
    
    # --- MATHEMATICAL TIME SLICING ---
    total_source_duration = get_video_duration(local_video_path)
    total_scenes = scene.get("total_scenes", 1)
    scene_num = int(scene.get("scene_number", 1))
    
    slice_duration = total_source_duration / total_scenes if total_scenes > 0 else total_source_duration
    
    # Calculate the exact midpoint of this scene's structural slice in the original video
    target_time = ((scene_num - 1) * slice_duration) + (slice_duration / 2.0)
    print(f"[EXTRACTING] Scene {scene_num}/{total_scenes} -> Target Timecode: {target_time:.2f}s (Source Length: {total_source_duration:.2f}s)")
    # ---------------------------------
    
    frame_path = extract_frame(local_video_path, target_time)
    
    if not frame_path:
        scene["generated_image_url"] = None
        return scene
        
    base_frame_url = upload_video(frame_path)
    
    image_urls = [base_frame_url]
    visual_text = scene['visual_prompt']
    
    # Strict Anti-Text Shielding
    design_prompt = (
        f"PHOTOREALISTIC COMPOSITION. CRITICAL: DO NOT GENERATE ANY TEXT, WORDS, OR TYPOGRAPHY.\n"
        f"SCENE: {visual_text}\n"
        f"- #Image1 is the base environment.\n"
    )
    
    # Smart Asset Binding Logic
    assets_injected = False
    if asset_urls:
        for idx, url in enumerate(asset_urls):
            asset_tag = f"#Image{idx + 2}" # Offset by 2 since Image1 is the base frame
            if asset_tag.lower() in visual_text.lower():
                image_urls.append(url)
                design_prompt += f"- {asset_tag} is a core visual subject. Seamlessly integrate its exact features into the scene.\n"
                assets_injected = True
                
    if not assets_injected:
        design_prompt += "Enhance this scene to be highly photorealistic while maintaining the core base composition. Ignore external tags if none apply."
    
    args = {"prompt": design_prompt, "image_urls": image_urls, "aspect_ratio": aspect_ratio}
    
    try:
        result = fal_client.subscribe(IMAGE_EDIT_MODEL, arguments=args)
        scene["generated_image_url"] = result["images"][0]["url"] if result and "images" in result else None
    except Exception as e:
        print(f"Error on Scene {scene['scene_number']}: {e}")
        scene["generated_image_url"] = None 
        
    return scene

def generate_storyboard_frames(extracted_text: str, uploaded_assets: List[str], local_video_path: str, aspect_ratio: str = "9:16") -> List[Dict]:
    """Pass 3 Orchestrator: Computes smart bindings and mathematical time slicing."""
    scenes = parse_extracted_scenes(extracted_text)
    if not scenes:
        raise ValueError("Could not parse scenes. Check VLM output format.")
        
    asset_urls = []
    for asset in uploaded_assets:
        if asset:
            url = upload_video(asset) if not asset.startswith("http") else asset
            asset_urls.append(url)
            
    storyboard_results = []
    
    # Run API calls concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for scene in scenes:
            futures.append(executor.submit(_process_single_scene, scene, asset_urls, local_video_path, aspect_ratio))
            
        for future in concurrent.futures.as_completed(futures):
            storyboard_results.append(future.result())
            
    # CRITICAL: Re-sort the flipbook into sequence since threads finish out of order
    storyboard_results.sort(key=lambda x: int(x["scene_number"]))
    return storyboard_results