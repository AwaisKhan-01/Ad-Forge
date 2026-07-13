"""
Orchestrates video generation using a Temporally Isolated Hard-Cut Strategy.
Features Proportional Time Allocation, T5-Optimized Prompts, and PARALLEL Video Rendering.
Located in: app/new_pipeline/generator.py
"""
import os
import re
import uuid
import requests
import fal_client
import concurrent.futures
from typing import List, Dict

from app.pipeline.model_registry import MODELS
from app.pipeline.cost_tracker import CostTracker
from app.pipeline.frame_extractor import extract_last_frame # Kept in case of future fallback needs
from app.pipeline.uploader import upload_video
from app.new_pipeline.stitcher import stitch_segments_natively

SEGMENTS_DIR = "segments"

def get_semantic_duration(ts_str: str) -> float:
    try:
        parts = ts_str.replace(" ", "").split('-')
        def time_to_sec(t_str):
            if ':' in t_str:
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        start_sec = time_to_sec(parts[0])
        if len(parts) > 1:
            return max(1.0, time_to_sec(parts[1]) - start_sec)
        return 3.0 
    except Exception:
        return 3.0

def plan_dynamic_segments(storyboard_frames: List[Dict], total_target_duration: int) -> List[float]:
    if not storyboard_frames:
        return []
    
    semantic_durations = [get_semantic_duration(s.get("timestamp", "")) for s in storyboard_frames]
    total_semantic_time = sum(semantic_durations)
    
    if total_semantic_time <= 0:
        base = float(total_target_duration) / len(storyboard_frames)
        return [base] * len(storyboard_frames)
        
    allocated_durations = []
    for sem_dur in semantic_durations:
        weight = sem_dur / total_semantic_time
        allocated_durations.append(weight * float(total_target_duration))
        
    return allocated_durations

def sanitize_video_prompt(raw_desc: str) -> str:
    clean = raw_desc.replace('"', '').replace("'", "")
    bad_words = ["cut to", "cut", "fade", "transition", "audio", "music", "voiceover", "scene", "visual", "transform", "override"]
    for word in bad_words:
        clean = re.sub(rf'(?i)\b{word}\b', '', clean)
    clean = clean.strip()
    if clean and not clean.endswith('.'):
        clean += '.'
    return clean

def _download_result(url: str) -> str:
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    path = os.path.join(SEGMENTS_DIR, f"{uuid.uuid4()}.mp4")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f: 
            f.write(r.content)
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to download video chunk: {e}")

# --- SURGICAL FIX: Extracted Worker Function for Threading ---
def _process_single_video_segment(idx: int, scene: Dict, seg_duration: float, total_scenes: int, 
                                  final_instructions: str, model: dict, aspect_ratio: str, 
                                  resolution: str, production_type: str) -> Dict:
    raw_v_desc = scene.get("visual_prompt", "")
    raw_v_desc = re.sub(r"(?i)\s*\(?#asset\d+\)?", "", raw_v_desc)
    
    v_desc = sanitize_video_prompt(raw_v_desc)
    m_desc = sanitize_video_prompt(scene.get("motion", ""))
    img_url = scene.get("generated_image_url")
    scene_audio = scene.get("audio", "").strip()
    
    isolated_prompt = f"{v_desc} Camera motion: {m_desc} Cinematic lighting, clean professional render."
    
    if scene_audio and "On-Screen Actor" in production_type:
        audio_word_count = len(scene_audio.split())
        if "(" in scene_audio and ")" in scene_audio and audio_word_count < 6:
            required_audio_time = 0.0
        else:
            required_audio_time = (audio_word_count * 0.4) + 1.0
            
        true_duration = max(seg_duration, required_audio_time)
        
        if "music" in scene_audio.lower() or "score" in scene_audio.lower() or scene_audio in ["", "..."]:
            isolated_prompt += " Subject is silent, natural ambient movement. No text on screen."
        else:
            safe_audio = scene_audio.replace('"', '').replace("'", "")
            isolated_prompt += (
                f' The subject is speaking exactly these words: "{safe_audio}". '
                f'CRITICAL TIMING: Start the video clip cleanly without the text. '
                f'The typography must dynamically animate and reveal itself ON SCREEN ONLY as the subject begins speaking.'
            )
        
    elif scene_audio and "Motion Graphic" in production_type:
        true_duration = seg_duration
        isolated_prompt += " Smooth kinetic motion. Text animates dynamically."
    else:
        true_duration = seg_duration
            
    if idx == total_scenes - 1 and final_instructions:
        isolated_prompt += f" {sanitize_video_prompt(final_instructions)}"

    safe_integer_duration = max(3, int(round(true_duration)))

    args = {
        "prompt": isolated_prompt.strip(), 
        "duration": safe_integer_duration, 
        "aspect_ratio": aspect_ratio
    }
    
    # We now strictly rely on the frame_designer composite image, no legacy continuity fallbacks
    if img_url:
        args["image_url"] = img_url
        args["image_urls"] = [img_url]

    if model.get("input_type") == "video": 
        args["resolution"] = resolution
    
    print(f"\n      [DISPATCHING] Scene {idx+1}/{total_scenes} -> {safe_integer_duration}s requested")
    
    result = fal_client.subscribe(model["endpoint"], arguments=args)
    out_url = result.get("video", {}).get("url")
    
    if not out_url: 
        raise RuntimeError(f"Model failed to return video URL on scene {idx+1}.")
        
    local_path = _download_result(out_url)
    print(f"      [COMPLETED] Scene {idx+1}")
    
    return {
        "idx": idx,
        "path": local_path,
        "duration": safe_integer_duration
    }

def generate_final_video(storyboard_frames: List[Dict], final_instructions: str, model_id: str, 
                         cost_tracker: CostTracker, total_duration: int, aspect_ratio: str, 
                         resolution: str, raw_script_text: str, production_type: str = "Narrative") -> str:
    
    model = MODELS[model_id]
    total_scenes = len(storyboard_frames)
    
    if not storyboard_frames: 
        raise RuntimeError("No valid scenes found in the structured storyboard.")
        
    scene_durations = plan_dynamic_segments(storyboard_frames, total_duration)

    # --- SURGICAL FIX: Parallel Execution Pipeline ---
    print("\n[STORYBOARD ENGINE] Initializing Parallel Video Rendering Matrix...")
    segment_results = []
    
    # max_workers=5 protects against aggressive rate-limiting from the video model endpoint
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, scene in enumerate(storyboard_frames):
            futures.append(executor.submit(
                _process_single_video_segment, 
                idx, scene, scene_durations[idx], total_scenes, 
                final_instructions, model, aspect_ratio, resolution, production_type
            ))
            
        for future in concurrent.futures.as_completed(futures):
            segment_results.append(future.result())
            
    # Sort the multithreaded results back into strict chronological order based on 'idx'
    segment_results.sort(key=lambda x: x["idx"])
    
    # Safely unpack the paths and update the cost tracker in sequence
    generated_paths = []
    for res in segment_results:
        generated_paths.append(res["path"])
        cost_tracker.add_segment(model_id, res["duration"], resolution)
            
    print("\n[STITCHING] Compiling isolated scenes natively into master timeline...")
    final_video_path = stitch_segments_natively(generated_paths)
    
    return final_video_path