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

from app.new_pipeline.model_registry import MODELS
from app.new_pipeline.cost_tracker import CostTracker
from app.pipeline.frame_extractor import extract_last_frame 
from app.new_pipeline.uploader import upload_video
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

    allocated_durations = []

    for s in storyboard_frames:
        sem_dur = get_semantic_duration(s.get("timestamp", ""))
        
        # --- SURGICAL FIX: Calculate pacing ONLY against Spoken Dialogue ---
        scene_audio = s.get("spoken_dialogue", "").strip()
        word_count = len(scene_audio.split()) if scene_audio else 0
        
        audio_time = (word_count * 0.6) + 2.0 if word_count > 0 else 0
        absolute_minimum = max(sem_dur, audio_time)
        
        base_min = 5.0 if word_count > 0 else 3.0
        absolute_minimum = max(base_min, absolute_minimum)
        allocated_durations.append(absolute_minimum)
        
    current_total = sum(allocated_durations)
    
    if current_total < total_target_duration and current_total > 0:
        expansion_factor = total_target_duration / current_total
        allocated_durations = [d * expansion_factor for d in allocated_durations]
        
    return allocated_durations

def sanitize_video_prompt(raw_desc: str) -> str:
    clean = raw_desc.replace('"', '').replace("'", "")
    
    safety_replacements = {
        "ripped": "distressed",
        "naked": "bare",
        "shoot": "film",
        "shot": "filmed",
        "kill": "end",
        "blood": "red liquid",
        "bomb": "burst",
        "explosive": "dynamic",
        "gun": "prop",
        "weapon": "prop"
    }
    
    for bad_word, safe_word in safety_replacements.items():
        clean = re.sub(rf'(?i)\b{bad_word}\b', safe_word, clean)
        
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

def _process_single_video_segment(idx: int, scene: Dict, final_duration: int, total_scenes: int, 
                                  final_instructions: str, model: dict, aspect_ratio: str, 
                                  resolution: str, production_type: str) -> Dict:
    raw_v_desc = scene.get("visual_prompt", "")
    
    raw_v_desc = re.sub(r"(?i)\s*\(?#asset\d+\)?", " <IMAGE_REF_0> ", raw_v_desc)
    
    v_desc = sanitize_video_prompt(raw_v_desc)
    m_desc = sanitize_video_prompt(scene.get("motion", ""))
    img_url = scene.get("generated_image_url")
    
    # --- SURGICAL FIX: Fetch separated dialogue and sfx values ---
    scene_dialogue = scene.get("spoken_dialogue", "").strip()
    scene_sfx = scene.get("sfx_music", "").strip()
    
    isolated_prompt = f"{v_desc} Camera motion: {m_desc} Cinematic lighting, clean professional render."
    
    if scene_dialogue and "On-Screen Actor" in production_type:
        safe_audio = scene_dialogue.replace('"', '').replace("'", "")
        isolated_prompt += (
            f' The subject is speaking exactly these words: "{safe_audio}". '
            f'Action begins IMMEDIATELY. The subject starts speaking on the very first frame. Do not delay.'
        )
    elif not scene_dialogue and "On-Screen Actor" in production_type:
        isolated_prompt += " Subject is silent, natural ambient movement. No text on screen."
        
    if scene_dialogue and "Motion Graphic" in production_type:
        isolated_prompt += " Smooth kinetic motion. Text animates dynamically."
            
    if idx == total_scenes - 1 and final_instructions:
        isolated_prompt += f" {sanitize_video_prompt(final_instructions)}"

    final_image_target = img_url if img_url else "https://storage.googleapis.com/falserverless/example_inputs/veo31-r2v-input-1.png"
    
    if not img_url:
        print(f"      [SYSTEM WARNING] Scene {idx+1} is missing a generated_image_url. Injecting a fallback anchor to prevent API validation crash.")

    args = {
        "prompt": isolated_prompt.strip(), 
        "duration": final_duration, 
        "aspect_ratio": aspect_ratio,
        "image_urls": [final_image_target]
    }
    
    args["image_url"] = final_image_target

    if model.get("input_type") == "video": 
        args["resolution"] = resolution
    
    print(f"\n      [DISPATCHING] Scene {idx+1}/{total_scenes} -> {final_duration}s requested")
    print(f"      [PROMPT SENT]: {args['prompt']}")
    
    result = fal_client.subscribe(model["endpoint"], arguments=args)
    out_url = result.get("video", {}).get("url")
    
    if not out_url: 
        raise RuntimeError(f"Model failed to return video URL on scene {idx+1}.")
        
    local_path = _download_result(out_url)
    print(f"      [COMPLETED] Scene {idx+1}")
    
    return {
        "idx": idx,
        "path": local_path,
        "duration": final_duration
    }

def generate_final_video(storyboard_frames: List[Dict], final_instructions: str, model_id: str, 
                         cost_tracker: CostTracker, total_duration: int, aspect_ratio: str, 
                         resolution: str, raw_script_text: str, production_type: str = "Narrative") -> str:
    
    model = MODELS[model_id]
    total_scenes = len(storyboard_frames)
    
    if not storyboard_frames: 
        raise RuntimeError("No valid scenes found in the structured storyboard.")
        
    raw_durations = plan_dynamic_segments(storyboard_frames, total_duration)

    final_clamped_durations = []
    print("\n========================================================")
    print("📋 AD-FORGE PIPELINE: PRE-FLIGHT PACING ANALYSIS")
    print("========================================================")
    
    for idx, scene in enumerate(storyboard_frames):
        scene_dialogue = scene.get("spoken_dialogue", "").strip()
        word_count = len(scene_dialogue.split()) if scene_dialogue else 0
        has_speech = bool(word_count > 0 and "On-Screen Actor" in production_type)
        
        floor_limit = 5 if has_speech else 3
        clamped_int = max(floor_limit, min(10, int(round(raw_durations[idx]))))
        final_clamped_durations.append(clamped_int)
        
        print(f"Scene {idx+1}:")
        print(f"  └─ Spoken Words: {word_count}")
        print(f"  └─ Floor Limit Enforced: {floor_limit}s")
        print(f"  └─ FINAL BUDGET ASSIGNED: {clamped_int} seconds")
        print(f"  └─ Target Script lines: \"{scene_dialogue[:60]}...\"")
        print("-" * 40)
        
    projected_total = sum(final_clamped_durations)
    print(f"Projected Total Sequence Length: {projected_total} seconds")
    print("========================================================\n")

    print("[STORYBOARD ENGINE] Initializing Parallel Video Rendering Matrix...")
    segment_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, scene in enumerate(storyboard_frames):
            futures.append(executor.submit(
                _process_single_video_segment, 
                idx, scene, final_clamped_durations[idx], total_scenes, 
                final_instructions, model, aspect_ratio, resolution, production_type
            ))
            
        for future in concurrent.futures.as_completed(futures):
            segment_results.append(future.result())
            
    segment_results.sort(key=lambda x: x["idx"])
    
    generated_paths = []
    for res in segment_results:
        generated_paths.append(res["path"])
        cost_tracker.add_segment(model_id, res["duration"], resolution)
            
    print("\n[STITCHING] Compiling isolated scenes natively into master timeline...")
    final_video_path = stitch_segments_natively(generated_paths)
    
    return final_video_path