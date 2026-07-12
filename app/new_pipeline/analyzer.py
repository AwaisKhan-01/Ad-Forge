"""
Implements an Agentic QA Architecture with Pydantic Strict Typing.
Features FFmpeg-Driven Cut-Rate Density Calculation and Narrative Compression.
Located in: app/new_pipeline/analyzer.py
"""
import json
import re
import subprocess
import fal_client
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from app.config import NEMOTRON_MODEL

SYSTEM_PROMPT = "You are a senior video advertising director. You communicate ONLY in valid, strictly formatted JSON."

# ==========================================
# 1. PYDANTIC SCHEMAS (THE BOUNCERS)
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
# 2. ROBUST JSON EXTRACTOR
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
            raise RuntimeError(f"Extracted string is still invalid JSON: {e}")
            
    raise RuntimeError("Could not locate any valid JSON structure in the Agent's response.")

# ==========================================
# 3. FFMPEG PACING ALGORITHM
# ==========================================
def _calculate_dynamic_pacing(video_url: str, target_duration: int) -> int:
    """
    Uses FFmpeg to mathematically calculate the Cut-Rate Density of the source video
    and projects it onto the target duration with strict safety clamps.
    """
    print("\n[SYSTEM] Scanning reference video to calculate Cut-Rate Density...")
    try:
        # 1. Get original video duration
        dur_cmd = ["ffprobe", "-v", "error", "-show_entries",
                   "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_url]
        dur_output = subprocess.check_output(dur_cmd, text=True, timeout=30).strip()
        original_duration = float(dur_output)
        
        # 2. Scan for cuts (gt(scene,0.3) is standard 30% pixel change threshold)
        # Limit scan to max 60 seconds to prevent massive processing overhead
        scan_time = min(60.0, original_duration)
        scene_cmd = [
            "ffmpeg", "-ss", "0", "-t", str(scan_time), "-i", video_url,
            "-filter:v", "select='gt(scene,0.3)',showinfo", "-f", "null", "-"
        ]
        # FFmpeg outputs showinfo to stderr
        scene_output = subprocess.run(scene_cmd, capture_output=True, text=True, timeout=60).stderr
        
        # Each camera cut generates a 'Parsed_showinfo' log
        cuts_found = scene_output.count("Parsed_showinfo")
        scenes_in_scan = cuts_found + 1 
        
        # 3. Calculate Density
        density_sec_per_scene = scan_time / scenes_in_scan
        print(f"[*] Found {cuts_found} cuts in {scan_time}s (Density: 1 scene every {density_sec_per_scene:.2f}s)")
        
        # 4. Project onto target duration
        projected_scenes = target_duration / density_sec_per_scene
        
    except Exception as e:
        print(f"[SYSTEM WARNING] FFmpeg scan failed or timed out: {e}. Falling back to standard 4s pacing.")
        density_sec_per_scene = 4.0
        projected_scenes = target_duration / density_sec_per_scene

    # 5. Apply the Safety Clamps
    # Rule: Absolute minimum is 3s per scene. Absolute maximum is 8s per scene.
    max_allowed_scenes = max(1, target_duration // 3)
    min_allowed_scenes = max(1, target_duration // 8)
    
    final_scenes = max(min_allowed_scenes, min(int(round(projected_scenes)), max_allowed_scenes))
    print(f"[*] Pacing Engine Decision: {final_scenes} scenes required for a {target_duration}s commercial.")
    
    return final_scenes

# ==========================================
# 4. AGENTIC FUNCTIONS
# ==========================================
def _generate_initial_draft(video_url: str, target_duration: int, exact_scenes: int, creative_directive: str, production_type: str) -> dict:
    prompt = f"""Analyze the ENTIRE reference video from start to finish and generate a new, highly condensed storyboard adapted for a {target_duration}-second {production_type} commercial.

CRITICAL USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"

RULES:
1. NARRATIVE COMPRESSION (CRITICAL): Compress the ENTIRE storyline into exactly {exact_scenes} scenes. Strategically select key moments from the BEGINNING, MIDDLE, and END of the source video.
2. TIMESTAMPS: Output the true source timecodes of where these key moments actually occur in the original video.
3. SCENE COUNT: You MUST output EXACTLY {exact_scenes} scenes in your JSON array. 
4. COLOR & STYLING: If the User Directive specifies colors or subjects, explicitly write those EXACT words into the 'visual_prompt'.
5. SINGLE SHOTS: 'visual_prompt' must describe only ONE camera shot per scene.
6. ESCAPING: NEVER use unescaped double quotes inside your text values. Use single quotes.

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
    result = fal_client.subscribe(NEMOTRON_MODEL, arguments={"prompt": prompt, "system_prompt": SYSTEM_PROMPT, "video_url": video_url, "max_tokens": 4096, "temperature": 0.2})
    return _extract_clean_json(result.get("output", ""))

def _critique_and_revise(draft_json: dict, creative_directive: str, target_duration: int, exact_scenes: int, video_url: str) -> dict:
    critique_prompt = f"""You are an elite QA Agent. Review this draft JSON storyboard against the User's strict directive and narrative compression rules.
    
USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"
TARGET SCENE COUNT: EXACTLY {exact_scenes} scenes.
DRAFT JSON: {json.dumps(draft_json)}

TASK: 
1. Did the draft capture the FULL narrative arc (Beginning, Middle, End) of the original video? If it cut off early, rewrite the scenes to span the entire timeline.
2. Does the draft have EXACTLY {exact_scenes} scenes? If it has more, delete the extra scenes.
3. Did the draft successfully incorporate styling/colors requested in the User Directive? If not, inject them.

CRITICAL JSON RULES:
1. Return ONLY valid JSON. No conversational text.
2. ESCAPING: NEVER use unescaped double quotes inside your text values. Use single quotes.
"""
    result = fal_client.subscribe(NEMOTRON_MODEL, arguments={"prompt": critique_prompt, "system_prompt": SYSTEM_PROMPT, "video_url": video_url, "max_tokens": 4096, "temperature": 0.1})
    return _extract_clean_json(result.get("output", ""))

# ==========================================
# 5. THE EVALUATOR-OPTIMIZER LOOP
# ==========================================
def analyze_and_extract_scenes(video_url: str, target_duration: int, creative_directive: str = "", production_type: str = "Narrative") -> str:
    # Use the FFmpeg Algorithmic Pacing Engine
    exact_scenes = _calculate_dynamic_pacing(video_url, target_duration)
    
    print(f"\n[AGENT: DIRECTOR] Drafting {exact_scenes}-scene structured storyboard spanning the entire source video...")
    
    max_retries = 3
    attempt = 0
    final_storyboard = None
    
    while attempt < max_retries:
        try:
            draft_data = _generate_initial_draft(video_url, target_duration, exact_scenes, creative_directive, production_type)
            
            print(f"[AGENT: CRITIC] Evaluating draft for Narrative Compression and User Directives...")
            draft_data = _critique_and_revise(draft_data, creative_directive, target_duration, exact_scenes, video_url) 
            
            if len(draft_data.get("scenes", [])) > exact_scenes:
                print(f"[SYSTEM] Trimming {len(draft_data['scenes']) - exact_scenes} hallucinated extra scenes...")
                draft_data["scenes"] = draft_data["scenes"][:exact_scenes]

            final_storyboard = StoryboardOutput(**draft_data)
            print("[SYSTEM] JSON Schema Validation Passed. Storyboard locked.")
            break
            
        except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
            print(f"[SYSTEM WARNING] Agent output failed schema validation (Attempt {attempt + 1}/{max_retries}). Error: {e}")
            attempt += 1
            if attempt == max_retries:
                raise RuntimeError(f"Agentic loop failed to produce valid JSON after {max_retries} attempts.")

    output_str = f"=== STRATEGY ===\n{final_storyboard.macro_strategy}\n\n=== STORYBOARD ===\n[MASTER AUDIO]\n{final_storyboard.master_audio}\n\n"
    
    for scene in final_storyboard.scenes:
        output_str += f"SCENE {scene.scene_number}\n"
        output_str += f"Timestamp: {scene.timestamp}\n"
        output_str += f"Visual: {scene.visual_prompt}\n"
        output_str += f"Motion: {scene.motion}\n"
        output_str += f"Audio: {scene.audio}\n\n"

    return output_str.strip()