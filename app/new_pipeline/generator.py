"""
Orchestrates video generation using a Temporally Isolated Hard-Cut Strategy.
Features Proportional Time Allocation and Audio-Driven Duration Padding.
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
    """Calculates the physical duration in seconds from a semantic timecode string."""
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
    """Allocates video duration proportionally based on the VLM's source timestamps."""
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

def sanitize_prompt(raw_desc: str) -> str:
    """Aggressively strips text instructions and editing vocabulary."""
    clean = re.sub(r'(?i)text[\s:]+.*?(?=\.|$)', '', raw_desc)
    clean = re.sub(r'".*?"', '', clean) 
    bad_words = ["cut to", "cut", "fade", "transition"]
    for word in bad_words:
        clean = re.sub(rf'(?i)\b{word}\b', '', clean)
    return clean.strip()

def _download_result(url: str) -> str:
    """Downloads the generated video chunk for continuity extraction."""
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
        raise RuntimeError("No valid scenes found in the storyboard.")
        
    scene_durations = plan_dynamic_segments(storyboard_frames, total_duration)
    is_motion_graphic = "Motion Graphic" in production_type
         
    match_music = re.search(r"(?:Music|Music Track|Music & SFX):\s*(.*?)(?=\n---|\nTimeline:|\nSCENE|\Z)", raw_script_text, re.DOTALL | re.IGNORECASE)
    music_style = match_music.group(1).strip() if match_music else "Upbeat commercial track."

    generated_paths = []
    continuity_url = None
    
    for idx, scene in enumerate(storyboard_frames):
        seg_duration = scene_durations[idx]
        
        base_cinematography = "High-end cinematic rendering, 8k resolution, professional lighting, highly detailed. CRITICAL: DO NOT RENDER WRITTEN TEXT ON SCREEN."
        
        v_desc = sanitize_prompt(scene.get("visual_prompt", ""))
        m_desc = sanitize_prompt(scene.get("motion", ""))
        img_url = scene.get("generated_image_url")
        
        packed_images = []
        current_image_idx = 1
        
        if continuity_url:
            packed_images.append(continuity_url)
            image_binding_text = f"The environment must seamlessly transition from the visual layout of #Image1. "
            current_image_idx = 2
        else:
            image_binding_text = ""
            
        if img_url:
            packed_images.append(img_url)
            image_binding_text += f"The main visual composition must strictly match the features and subject placement of #Image{current_image_idx}. "
            
        isolated_prompt = f"{base_cinematography} {image_binding_text}SCENE ACTION: {v_desc}. CAMERA MOVEMENT: {m_desc}. "
        
        scene_audio = scene.get("audio", "").strip()
        
        # --- THE FIX: AUDIO-DRIVEN DURATION PADDING ---
        # Calculates exactly how long it takes a human to speak the assigned text
        if scene_audio and "On-Screen Actor" in production_type:
            audio_word_count = len(scene_audio.split())
            
            # Ignores placeholder text like "(No specific dialogue)"
            if "(" in scene_audio and ")" in scene_audio and audio_word_count < 6:
                required_audio_time = 0.0
            else:
                # 0.4 seconds per word (2.5 words/sec) + 1 second of natural breathing room
                required_audio_time = (audio_word_count * 0.4) + 1.0
                
            # The clip gets whichever is longer: the visual math, or the time required to speak
            true_duration = max(seg_duration, required_audio_time)
            
            label_intent = "dialogue performance"
            isolated_prompt += f"Visual tracking and actor lip movement must align perfectly and continuously with this {label_intent}: {scene_audio}. "
            
        elif scene_audio and is_motion_graphic:
            true_duration = seg_duration
            isolated_prompt += f"Kinetic motion beats synchronized to this SFX profile: '{scene_audio}'. "
        else:
            true_duration = seg_duration
        # ----------------------------------------------
                
        isolated_prompt += f"GLOBAL AUDIO BACKDROP: Background music is '{music_style}'."
        
        if idx == total_scenes - 1 and final_instructions:
            isolated_prompt += f" FINAL OUTRO DIRECTIVE: {final_instructions}"

        # API Safety Clamp - Final override to ensure it never crashes
        safe_integer_duration = max(3, int(round(true_duration)))

        args = {
            "prompt": isolated_prompt.strip(), 
            "duration": safe_integer_duration, 
            "aspect_ratio": aspect_ratio,
            "image_urls": packed_images[:model.get("max_images", 9)]
        }
        
        if model.get("input_type") == "video": 
            args["resolution"] = resolution
        
        print(f"\n--- Generating Isolated Scene {idx+1}/{total_scenes} ({safe_integer_duration}s requested) ---")
        print(f"Executing Prompt: {args['prompt']}")
        
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