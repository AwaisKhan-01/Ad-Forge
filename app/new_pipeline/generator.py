"""
Orchestrates video generation using a Temporally Isolated Hard-Cut Strategy.
Features Proportional Time Allocation and T5-Optimized Natural Language Prompts.
Located in: app/new_pipeline/generator.py
"""
import os
import re
import uuid
import requests
import fal_client
from typing import List, Dict

from app.pipeline.model_registry import MODELS
from app.pipeline.cost_tracker import CostTracker
from app.pipeline.frame_extractor import extract_last_frame
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
    """Cleans natural language for T5 Video Encoders and removes breaking characters."""
    # 1. Remove quotes to prevent backend shell injection/parsing crashes
    clean = raw_desc.replace('"', '').replace("'", "")
    
    # 2. Remove UI/Agent logic words that confuse video models
    bad_words = ["cut to", "cut", "fade", "transition", "audio", "music", "voiceover", "scene", "visual", "transform", "override"]
    for word in bad_words:
        clean = re.sub(rf'(?i)\b{word}\b', '', clean)
        
    # 3. Ensure it ends with a period for natural language flow
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

def generate_final_video(storyboard_frames: List[Dict], final_instructions: str, model_id: str, 
                         cost_tracker: CostTracker, total_duration: int, aspect_ratio: str, 
                         resolution: str, raw_script_text: str, production_type: str = "Narrative") -> str:
    
    model = MODELS[model_id]
    total_scenes = len(storyboard_frames)
    
    if not storyboard_frames: 
        raise RuntimeError("No valid scenes found in the structured storyboard.")
        
    scene_durations = plan_dynamic_segments(storyboard_frames, total_duration)
    is_motion_graphic = "Motion Graphic" in production_type

    generated_paths = []
    continuity_url = None
    
    for idx, scene in enumerate(storyboard_frames):
        seg_duration = scene_durations[idx]
        
        # --- SURGICAL FIX: Strip the #Asset tracking tags before it hits the video T5 encoder ---
        raw_v_desc = scene.get("visual_prompt", "")
        raw_v_desc = re.sub(r"(?i)\s*\(?#asset\d+\)?", "", raw_v_desc)
        
        v_desc = sanitize_video_prompt(raw_v_desc)
        m_desc = sanitize_video_prompt(scene.get("motion", ""))
        img_url = scene.get("generated_image_url")
        scene_audio = scene.get("audio", "").strip()
        
        # --- SURGICAL FIX: Removed "highly detailed" poison tag. Use clean cinematic keywords. ---
        isolated_prompt = f"{v_desc} Camera motion: {m_desc} Cinematic lighting, clean professional render."
        
        if scene_audio and "On-Screen Actor" in production_type:
            audio_word_count = len(scene_audio.split())
            if "(" in scene_audio and ")" in scene_audio and audio_word_count < 6:
                required_audio_time = 0.0
            else:
                required_audio_time = (audio_word_count * 0.4) + 1.0
                
            true_duration = max(seg_duration, required_audio_time)
            
            # --- SURGICAL FIX: Bind the EXACT extracted text to the prompt, avoiding music tracks ---
            if "music" in scene_audio.lower() or "score" in scene_audio.lower() or scene_audio in ["", "..."]:
                isolated_prompt += " Subject is silent, natural ambient movement."
            else:
                safe_audio = scene_audio.replace('"', '').replace("'", "")
                isolated_prompt += f' The subject is speaking exactly these words: "{safe_audio}".'
            
        elif scene_audio and is_motion_graphic:
            true_duration = seg_duration
            isolated_prompt += " Smooth kinetic motion."
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
        
        primary_image = img_url or continuity_url
        if primary_image:
            args["image_url"] = primary_image
            args["image_urls"] = [primary_image]

        if model.get("input_type") == "video": 
            args["resolution"] = resolution
        
        print(f"\n--- Generating Isolated Scene {idx+1}/{total_scenes} ({safe_integer_duration}s requested) ---")
        print(f"Video LLM Prompt: {args['prompt']}")
        
        result = fal_client.subscribe(model["endpoint"], arguments=args)
        out_url = result.get("video", {}).get("url")
        
        if not out_url: 
            raise RuntimeError(f"Model failed to return video URL on scene {idx+1}.")
            
        local_path = _download_result(out_url)
        generated_paths.append(local_path)
        cost_tracker.add_segment(model_id, safe_integer_duration, resolution) 
        
        continuity_url = upload_video(extract_last_frame(local_path))
            
    print("\n[STITCHING] Compiling isolated scenes natively into master timeline...")
    final_video_path = stitch_segments_natively(generated_paths)
    
    return final_video_path