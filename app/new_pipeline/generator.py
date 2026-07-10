"""
Orchestrates video generation using a Temporally Isolated Hard-Cut Strategy.
Handles dynamic asset mapping and roots acoustic generation (SFX, Voiceover, Dialogue).
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

def plan_segments(total_duration: int, total_scenes: int) -> List[float]:
    """Divides the target duration evenly across the extracted scenes."""
    if total_scenes <= 0:
        return [float(total_duration)]
    base_duration = float(total_duration) / total_scenes
    return [base_duration] * total_scenes

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
        
    scene_durations = plan_segments(total_duration, total_scenes)
    is_motion_graphic = "Motion Graphic" in production_type
         
    # --- ACOUSTIC ROUTING PARSER (UPDATED) ---
    if is_motion_graphic:
        match_music = re.search(r"Music & SFX:\s*(.*?)(?=\n---|\nTimeline:|\nSCENE|\Z)", raw_script_text, re.DOTALL | re.IGNORECASE)
        music_style = match_music.group(1).strip() if match_music else "Energetic electronic motion graphics track."
        vo_words = []
        words_per_scene = 0
    else:
        # Check for Dialogue OR Voiceover tags seamlessly
        match_audio = re.search(r"(?:Dialogue|Voiceover):\s*(.*?)(?=\n(?:Music|Music Track):|\n---|\nTimeline:|\nSCENE|\Z)", raw_script_text, re.DOTALL | re.IGNORECASE)
        match_music = re.search(r"(?:Music|Music Track):\s*(.*?)(?=\n---|\nTimeline:|\nSCENE|\Z)", raw_script_text, re.DOTALL | re.IGNORECASE)
        
        full_audio_script = match_audio.group(1).strip() if match_audio else ""
        music_style = match_music.group(1).strip() if match_music else "Upbeat commercial music"
        
        vo_words = full_audio_script.split()
        words_per_scene = max(1, len(vo_words) // total_scenes) if total_scenes > 0 else 0
    # -----------------------------------------

    generated_paths = []
    continuity_url = None
    
    # --- TEMPORALLY ISOLATED GENERATION LOOP ---
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
        
        if is_motion_graphic:
            scene_sfx = scene.get("audio", "").strip()
            if scene_sfx:
                isolated_prompt += f"Kinetic motion beats synchronized to this SFX profile: '{scene_sfx}'. "
        else:
            s_word = idx * words_per_scene
            e_word = (idx + 1) * words_per_scene
            if idx == total_scenes - 1:
                e_word = len(vo_words)
                
            scene_vo = " ".join(vo_words[s_word:e_word]).strip()
            if scene_vo:
                # Dynamically tag behavior framework based on active delivery method
                label_intent = "dialogue performance" if "On-Screen Actor" in production_type else "narrative pacing focus"
                isolated_prompt += f"Visual tracking and actor lip movement must align with this {label_intent}: '{scene_vo}'. "
                
        isolated_prompt += f"GLOBAL AUDIO BACKDROP: Background music is '{music_style}'."
        
        if idx == total_scenes - 1 and final_instructions:
            isolated_prompt += f" FINAL OUTRO DIRECTIVE: {final_instructions}"

        args = {
            "prompt": isolated_prompt.strip(), 
            "duration": f"{seg_duration:.1f}", 
            "aspect_ratio": aspect_ratio,
            "image_urls": packed_images[:model.get("max_images", 9)]
        }
        
        if model.get("input_type") == "video": 
            args["resolution"] = resolution
        
        print(f"\n--- Generating Isolated Scene {idx+1}/{total_scenes} ({seg_duration:.1f}s) ---")
        print(f"Executing Prompt: {args['prompt']}")
        
        result = fal_client.subscribe(model["endpoint"], arguments=args)
        out_url = result.get("video", {}).get("url")
        
        if not out_url: 
            raise RuntimeError(f"Model failed to return video URL on scene {idx+1}.")
            
        local_path = _download_result(out_url)
        generated_paths.append(local_path)
        cost_tracker.add_segment(model_id, seg_duration, resolution)
        
        continuity_url = upload_video(extract_last_frame(local_path))
            
    print("\n[STITCHING] Compiling isolated scenes natively into master timeline...")
    final_video_path = stitch_segments_natively(generated_paths)
    
    return final_video_path