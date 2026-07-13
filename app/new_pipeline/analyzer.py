"""
Implements an Agentic QA Architecture with Pydantic Strict Typing.
Features Dynamic Asset Allocation to prevent global asset leaking and enforce static temporal constraints.
Located in: app/new_pipeline/analyzer.py
"""
import json
import re
import subprocess
from dotenv import load_dotenv

load_dotenv() 

import fal_client
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

SYSTEM_PROMPT = "You are a senior video advertising director. You communicate ONLY in valid, strictly formatted JSON."
VLM_ENDPOINT = "openrouter/router/video"
TARGET_VLM_MODEL = "google/gemini-3.1-pro-preview"

class Scene(BaseModel):
    scene_number: int = Field(..., description="The sequential number of the scene.")
    timestamp: str = Field(..., description="The exact source timecode, e.g., '00:00 - 00:03'")
    # SURGICAL FIX 1: Explicit Keyframe Targeting
    keyframe_time: str = Field(..., description="The EXACT single second (e.g., '00:02') where the visual_prompt is most perfectly visible.")
    visual_prompt: str = Field(..., description="ONE strictly static, frozen visual description. NO transitions or sequential actions.")
    motion: str = Field(..., description="Camera movement description.")
    audio: str = Field(..., description="Audio cues or dialogue for this scene.")
    assigned_assets: List[int] = Field(default=[], description="Indices of the user assets (e.g. [0] for Asset 0) that apply STRICTLY to this scene. Leave empty [] if none apply.")

class StoryboardOutput(BaseModel):
    macro_strategy: str = Field(..., description="The high-level marketing strategy and pacing calculation.")
    master_audio: str = Field(..., description="The global audio track or music style.")
    scenes: List[Scene] = Field(..., description="The list of perfectly calculated scenes.")

def _extract_clean_json(raw_text: str) -> dict:
    # SURGICAL FIX 2: Robust Extraction to bypass Gemini "Reasoning" blocks
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}') + 1
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No braces found")
            
        json_str = cleaned[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise RuntimeError(f"Could not locate valid JSON structure. Raw response: {cleaned[:100]}...")

def _calculate_dynamic_pacing(video_url: str, target_duration: int) -> int:
    print("\n[SYSTEM] Scanning reference video to calculate Cut-Rate Density...")
    try:
        dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_url]
        dur_output = subprocess.check_output(dur_cmd, text=True, timeout=30).strip()
        original_duration = float(dur_output)
        
        scan_time = min(60.0, original_duration)
        scene_cmd = ["ffmpeg", "-ss", "0", "-t", str(scan_time), "-i", video_url, "-filter:v", "select='gt(scene,0.3)',showinfo", "-f", "null", "-"]
        scene_output = subprocess.run(scene_cmd, capture_output=True, text=True, timeout=60).stderr
        
        cuts_found = scene_output.count("Parsed_showinfo")
        scenes_in_scan = cuts_found + 1 
        
        density_sec_per_scene = scan_time / scenes_in_scan
        print(f"[*] Found {cuts_found} cuts in {scan_time}s (Density: 1 scene every {density_sec_per_scene:.2f}s)")
        projected_scenes = target_duration / density_sec_per_scene
    except Exception as e:
        print(f"[SYSTEM WARNING] FFmpeg scan failed: {e}. Falling back to 4s pacing.")
        projected_scenes = target_duration / 4.0

    max_allowed_scenes = max(1, target_duration // 3)
    min_allowed_scenes = max(1, target_duration // 8)
    final_scenes = max(min_allowed_scenes, min(int(round(projected_scenes)), max_allowed_scenes))
    print(f"[*] Pacing Engine Decision: {final_scenes} scenes required.")
    return final_scenes

def _generate_initial_draft(video_url: str, target_duration: int, exact_scenes: int, creative_directive: str, production_type: str, asset_instructions: str) -> dict:
    max_words = int(target_duration * 2.5)
    
    prompt = f"""Analyze the ENTIRE reference video and generate a highly condensed storyboard for a {target_duration}-second {production_type} commercial.

CRITICAL USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"

USER ASSET INSTRUCTIONS:
{asset_instructions if asset_instructions else 'No custom production assets uploaded.'}

RULES:
1. NARRATIVE COMPRESSION: Compress the storyline into exactly {exact_scenes} scenes. Select moments from BEGINNING, MIDDLE, and END.
2. NATIVE AUDIO CONTINUITY: Listen to the speech, spoken dialogue, and voiceover natively present in the video file. Edit and condense the actual spoken dialogue into a seamless, continuous script (maximum {max_words} words total).
3. MANDATORY ASSET MAPPING (CRITICAL): Read the 'USER ASSET INSTRUCTIONS'. You MUST replace the original subjects or objects in the video with the requested assets. If the user says "Asset 0 is the main person", you MUST write "#Asset0" instead of "man" or "woman" in the 'visual_prompt', AND you MUST put the integer 0 in the 'assigned_assets' array (e.g., [0]) for EVERY scene that character appears in. Do NOT leave the array empty if the subject is present.
4. FREEZE TIME (CRITICAL): The 'visual_prompt' MUST describe a STILL PHOTOGRAPH. NO TIMELINES. NO SEQUENCES. You are STRICTLY FORBIDDEN from using the word "then". Put all camera movement in the 'motion' field.
5. ESCAPING: NEVER use unescaped double quotes inside values. Use single quotes.
6. EXPLICIT KEYFRAME TARGETING: You must provide a 'keyframe_time' (e.g. '00:05'). This must be the EXACT second within the timestamp block that perfectly visually matches your static 'visual_prompt'.

OUTPUT FORMAT:
You must output pure JSON matching this schema:
{{
  "macro_strategy": "string",
  "master_audio": "string",
  "scenes": [
    {{
      "scene_number": 1,
      "timestamp": "00:00 - 00:03",
      "keyframe_time": "00:02",
      "visual_prompt": "string",
      "motion": "string",
      "audio": "string",
      "assigned_assets": []
    }}
  ]
}}
"""
    result = fal_client.subscribe(
        VLM_ENDPOINT, 
        arguments={
            "video_urls": [video_url],
            "prompt": prompt, 
            "system_prompt": SYSTEM_PROMPT, 
            "model": TARGET_VLM_MODEL,
            "temperature": 0.2,
            "reasoning": True
        }
    )
    return _extract_clean_json(result.get("output", ""))

def _critique_and_revise(draft_json: dict, creative_directive: str, exact_scenes: int, video_url: str, asset_instructions: str) -> dict:
    critique_prompt = f"""You are an elite QA Agent. Review this draft JSON storyboard against the video asset.
    
USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"
USER ASSET INSTRUCTIONS: "{asset_instructions if asset_instructions else 'None'}"
TARGET SCENE COUNT: EXACTLY {exact_scenes} scenes.
DRAFT JSON: {json.dumps(draft_json)}

QA TASK (FIX ERRORS DIRECTLY IN THE JSON):
1. EXACTLY {exact_scenes} scenes?
2. ENFORCE ASSET MAPPING: Did the draft assign assets to the correct scenes based on the USER ASSET INSTRUCTIONS? If the draft left 'assigned_assets' empty `[]` but the subject is clearly in the scene, YOU MUST FIX IT by adding the integer to the array (e.g., `[0]`) and injecting `#Asset0` into the visual_prompt.
3. ENFORCE TIME FREEZE: Read every 'visual_prompt'. If you see the word "then" or any sequential action (e.g., "walks in, then sits"), DELETE IT. Rewrite it to describe a single, frozen moment.
4. KEYFRAME INTEGRITY: Ensure 'keyframe_time' is present and accurately reflects a static moment within the scene's 'timestamp'.

CRITICAL: ONLY valid JSON. NEVER use unescaped double quotes.
"""
    result = fal_client.subscribe(
        VLM_ENDPOINT, 
        arguments={
            "video_urls": [video_url],
            "prompt": critique_prompt, 
            "system_prompt": SYSTEM_PROMPT, 
            "model": TARGET_VLM_MODEL,
            "temperature": 0.1,
            "reasoning": True
        }
    )
    return _extract_clean_json(result.get("output", ""))

def analyze_and_extract_scenes(video_url: str, target_duration: int, creative_directive: str = "", production_type: str = "Narrative", asset_instructions: str = "") -> str:
    exact_scenes = _calculate_dynamic_pacing(video_url, target_duration)
    
    print(f"\n[AGENT: DIRECTOR] Ingesting via OpenRouter {TARGET_VLM_MODEL}...")
    
    max_retries = 3
    attempt = 0
    final_storyboard = None
    
    while attempt < max_retries:
        try:
            draft_data = _generate_initial_draft(video_url, target_duration, exact_scenes, creative_directive, production_type, asset_instructions)
            print(f"[AGENT: CRITIC] Evaluating draft structure and asset isolation...")
            
            draft_data = _critique_and_revise(draft_data, creative_directive, exact_scenes, video_url, asset_instructions) 
            
            if len(draft_data.get("scenes", [])) > exact_scenes:
                draft_data["scenes"] = draft_data["scenes"][:exact_scenes]

            final_storyboard = StoryboardOutput(**draft_data)
            print("[SYSTEM] JSON Schema Validation Passed.")
            break
        except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
            print(f"[SYSTEM WARNING] Agent validation failed (Attempt {attempt + 1}/{max_retries}). Error: {e}")
            attempt += 1
            if attempt == max_retries:
                raise RuntimeError(f"Agentic loop failed after {max_retries} attempts.")

    output_str = f"=== STRATEGY ===\n{final_storyboard.macro_strategy}\n\n=== STORYBOARD ===\n[MASTER AUDIO]\n{final_storyboard.master_audio}\n\n"
    for scene in final_storyboard.scenes:
        # SURGICAL FIX 3: Inject the Keyframe line into the output string so frame_designer.py can regex it
        output_str += f"SCENE {scene.scene_number}\nTimestamp: {scene.timestamp}\nKeyframe: {scene.keyframe_time}\nVisual: {scene.visual_prompt}\nMotion: {scene.motion}\nAudio: {scene.audio}\nAssigned Assets: {json.dumps(scene.assigned_assets)}\n\n"
    return output_str.strip()