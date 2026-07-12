"""
Implements an Agentic QA Architecture with Pydantic Strict Typing.
Features FFmpeg Pacing and Pure OpenRouter Gemini 3.1 Video/Audio Ingestion.
Located in: app/new_pipeline/analyzer.py
"""
import json
import re
import subprocess
import fal_client
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

SYSTEM_PROMPT = "You are a senior video advertising director. You communicate ONLY in valid, strictly formatted JSON."
VLM_ENDPOINT = "openrouter/router/video"
TARGET_VLM_MODEL = "google/gemini-3.1-pro-preview"

# ==========================================
# 1. PYDANTIC SCHEMAS
# ==========================================
class Scene(BaseModel):
    scene_number: int = Field(..., description="The sequential number of the scene.")
    timestamp: str = Field(..., description="The exact source timecode, e.g., '00:00 - 00:03'")
    visual_prompt: str = Field(..., description="Strictly ONE static visual description. Must include color styling if requested.")
    motion: str = Field(..., description="Camera movement description.")
    audio: str = Field(..., description="Audio cues, dialogue, or SFX for this scene.")

class StoryboardOutput(BaseModel):
    macro_strategy: str = Field(..., description="The high-level marketing strategy and pacing calculation.")
    master_audio: str = Field(..., description="The global audio track or music style.")
    scenes: List[Scene] = Field(..., description="The list of perfectly calculated scenes.")

# ==========================================
# 2. JSON EXTRACTOR
# ==========================================
def _extract_clean_json(raw_text: str) -> dict:
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass
    if raw_text.strip().startswith("```"):
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Extracted string is invalid JSON: {e}")
    raise RuntimeError("Could not locate any valid JSON structure.")

# ==========================================
# 3. FFMPEG PACING ALGORITHM
# ==========================================
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

# ==========================================
# 4. AGENTIC FUNCTIONS (PURE NATIVE INGESTION)
# ==========================================
def _generate_initial_draft(video_url: str, target_duration: int, exact_scenes: int, creative_directive: str, production_type: str) -> dict:
    max_words = int(target_duration * 2.5)
    
    prompt = f"""Analyze the ENTIRE reference video (both visual sequence and native audio/speech track) and generate a highly condensed storyboard for a {target_duration}-second {production_type} commercial.

CRITICAL USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"

RULES:
1. NARRATIVE COMPRESSION: Compress the ENTIRE storyline into exactly {exact_scenes} scenes. Select moments from BEGINNING, MIDDLE, and END.
2. NATIVE AUDIO CONTINUITY (CRITICAL): Listen to the speech, spoken dialogue, and voiceover natively present in the video file. Do NOT guess words based on text flashing on screen. Edit and condense the actual spoken dialogue into a seamless, continuous script (maximum {max_words} words total). Distribute this script logically across your {exact_scenes} scenes.
3. TIMESTAMPS: Output true source timecodes of the visual moments.
4. SCENE COUNT: EXACTLY {exact_scenes} scenes in the JSON array.
5. COLOR & STYLING: Write exact requested styling into the 'visual_prompt'.
6. SINGLE SHOTS: 'visual_prompt' is ONE camera shot per scene.
7. ESCAPING: NEVER use unescaped double quotes inside values. Use single quotes.

OUTPUT FORMAT:
You must output pure JSON that exactly matches this schema:
{{
  "macro_strategy": "string",
  "master_audio": "string",
  "scenes": [
    {{
      "scene_number": 1,
      "timestamp": "00:00 - 00:03",
      "visual_prompt": "string",
      "motion": "string",
      "audio": "string"
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
            "temperature": 0.2
        }
    )
    return _extract_clean_json(result.get("output", ""))

def _critique_and_revise(draft_json: dict, creative_directive: str, exact_scenes: int, video_url: str) -> dict:
    critique_prompt = f"""You are an elite QA Agent. Review this draft JSON storyboard against the video asset.
    
USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"
TARGET SCENE COUNT: EXACTLY {exact_scenes} scenes.
DRAFT JSON: {json.dumps(draft_json)}

TASK: 
1. Capture FULL visual narrative arc?
2. EXACTLY {exact_scenes} scenes? Delete extra.
3. Incorporate styling/colors requested?
4. AUDIO CONTINUITY: Do the 'audio' fields flow like a continuous, logical script based on the actual spoken dialogue in the video without broken snippets?

CRITICAL: ONLY valid JSON. NEVER use unescaped double quotes.
"""
    result = fal_client.subscribe(
        VLM_ENDPOINT, 
        arguments={
            "video_urls": [video_url],
            "prompt": critique_prompt, 
            "system_prompt": SYSTEM_PROMPT, 
            "model": TARGET_VLM_MODEL,
            "temperature": 0.1
        }
    )
    return _extract_clean_json(result.get("output", ""))

# ==========================================
# 5. ORCHESTRATION LOOP
# ==========================================
def analyze_and_extract_scenes(video_url: str, target_duration: int, creative_directive: str = "", production_type: str = "Narrative") -> str:
    exact_scenes = _calculate_dynamic_pacing(video_url, target_duration)
    
    print(f"\n[AGENT: DIRECTOR] Ingesting via OpenRouter {TARGET_VLM_MODEL}...")
    
    max_retries = 3
    attempt = 0
    final_storyboard = None
    
    while attempt < max_retries:
        try:
            draft_data = _generate_initial_draft(video_url, target_duration, exact_scenes, creative_directive, production_type)
            print(f"[AGENT: CRITIC] Evaluating draft structure...")
            draft_data = _critique_and_revise(draft_data, creative_directive, exact_scenes, video_url) 
            
            if len(draft_data.get("scenes", [])) > exact_scenes:
                print(f"[SYSTEM] Trimming {len(draft_data['scenes']) - exact_scenes} extra scenes...")
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
        output_str += f"SCENE {scene.scene_number}\nTimestamp: {scene.timestamp}\nVisual: {scene.visual_prompt}\nMotion: {scene.motion}\nAudio: {scene.audio}\n\n"
    return output_str.strip()