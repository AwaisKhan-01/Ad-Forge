"""
Implements an Agentic QA Architecture with Pydantic Strict Typing.
Features a self-correcting Evaluator-Optimizer loop to prevent instruction dilution.
Located in: app/new_pipeline/analyzer.py
"""
import json
import re
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
    """Intelligently hunts and extracts JSON, ignoring LLM conversational hallucinations."""
    # First, try to load it straight away
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Next, strip common markdown wrappers
    if raw_text.strip().startswith("```"):
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
            
    # Finally, use regex to find the outermost brackets (the true JSON block)
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Extracted string is still invalid JSON: {e}")
            
    raise RuntimeError("Could not locate any valid JSON structure in the Agent's response.")

# ==========================================
# 3. AGENTIC FUNCTIONS
# ==========================================
def _generate_initial_draft(video_url: str, target_duration: int, creative_directive: str, production_type: str) -> dict:
    """The Director Agent: Attempts to write the perfect JSON storyboard."""
    
    prompt = f"""Analyze the reference video and generate a new storyboard adapted for a {target_duration}-second {production_type} commercial.

CRITICAL USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"

RULES:
1. COLOR & STYLING: If the User Directive specifies colors (e.g., "Pinkish Red"), you MUST explicitly write those EXACT color words into the 'visual_prompt' of any scene containing text or modified subjects.
2. TIMESTAMPS: Extract true timestamps from the reference video.
3. SINGLE SHOTS: 'visual_prompt' must describe only ONE camera shot per scene.
4. ESCAPING: NEVER use unescaped double quotes inside your text values. Use single quotes if you need to quote something inside a field.

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
        NEMOTRON_MODEL, 
        arguments={
            "prompt": prompt, 
            "system_prompt": SYSTEM_PROMPT, 
            "video_url": video_url, 
            "max_tokens": 4096, 
            "temperature": 0.2
        }
    )
    
    return _extract_clean_json(result.get("output", ""))

def _critique_and_revise(draft_json: dict, creative_directive: str, target_duration: int, video_url: str) -> dict:
    """The Critic Agent: Evaluates the draft against the user's explicit instructions."""
    
    if not creative_directive:
        return draft_json
        
    critique_prompt = f"""You are an elite QA Agent. Review this draft JSON storyboard against the User's strict directive.
    
USER DIRECTIVE: "{creative_directive}"
DRAFT JSON: {json.dumps(draft_json)}

TASK: Did the draft successfully incorporate the styling, colors, or subject changes requested in the User Directive into the 'visual_prompt' fields?
If it failed, rewrite the JSON to forcefully inject the user's specific words (e.g., color names, replacement subjects) into the relevant scenes.

CRITICAL JSON RULES:
1. Return ONLY valid JSON. No conversational text.
2. ESCAPING: NEVER use unescaped double quotes inside your text values. Use single quotes ('like this') if you need to quote something inside a field.
3. SCHEMA MATCH: Your output MUST perfectly match this structure:
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
        NEMOTRON_MODEL, 
        arguments={
            "prompt": critique_prompt, 
            "system_prompt": SYSTEM_PROMPT, 
            "video_url": video_url, 
            "max_tokens": 4096, 
            "temperature": 0.1
        }
    )
    
    return _extract_clean_json(result.get("output", ""))

# ==========================================
# 4. THE EVALUATOR-OPTIMIZER LOOP
# ==========================================
def analyze_and_extract_scenes(video_url: str, target_duration: int, creative_directive: str = "", production_type: str = "Narrative") -> str:
    print("\n[AGENT: DIRECTOR] Drafting initial structured storyboard...")
    max_retries = 3 # Increased to 3 to give the loop more breathing room
    attempt = 0
    final_storyboard = None
    
    while attempt < max_retries:
        try:
            draft_data = _generate_initial_draft(video_url, target_duration, creative_directive, production_type)
            
            if creative_directive:
                print("[AGENT: CRITIC] Evaluating draft against User Directive...")
                draft_data = _critique_and_revise(draft_data, creative_directive, target_duration, video_url) 
            
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