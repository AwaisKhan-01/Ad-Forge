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
# 3. AGENTIC FUNCTIONS
# ==========================================
def _generate_initial_draft(video_url: str, target_duration: int, exact_scenes: int, creative_directive: str, production_type: str) -> dict:
    """The Director Agent: Attempts to write the perfect JSON storyboard."""
    
    prompt = f"""Analyze the reference video and generate a new storyboard adapted for a {target_duration}-second {production_type} commercial.

CRITICAL USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"

RULES:
1. STRICT PACING LIMIT: A {target_duration}-second commercial can only physically hold {exact_scenes} scenes. You MUST output EXACTLY {exact_scenes} scenes in your JSON array. Do not describe the entire source video. Stop at scene {exact_scenes}.
2. COLOR & STYLING: If the User Directive specifies colors, you MUST explicitly write those EXACT color words into the 'visual_prompt'.
3. TIMESTAMPS: Extract true timestamps from the first {target_duration} seconds of the reference video.
4. SINGLE SHOTS: 'visual_prompt' must describe only ONE camera shot per scene.
5. ESCAPING: NEVER use unescaped double quotes inside your text values. Use single quotes if you need to quote something.

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

def _critique_and_revise(draft_json: dict, creative_directive: str, target_duration: int, exact_scenes: int, video_url: str) -> dict:
    """The Critic Agent: Evaluates the draft against the user's explicit instructions."""
    
    critique_prompt = f"""You are an elite QA Agent. Review this draft JSON storyboard against the User's strict directive and pacing constraints.
    
USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"
TARGET SCENE COUNT: EXACTLY {exact_scenes} scenes.
DRAFT JSON: {json.dumps(draft_json)}

TASK: 
1. Did the draft successfully incorporate the styling/colors requested in the User Directive? If not, rewrite the visual prompts to include them.
2. Does the draft have EXACTLY {exact_scenes} scenes? If it has more than {exact_scenes} scenes, you MUST delete the extra scenes from the JSON array.

CRITICAL JSON RULES:
1. Return ONLY valid JSON. No conversational text.
2. ESCAPING: NEVER use unescaped double quotes inside your text values. Use single quotes.
3. SCHEMA MATCH: Your output MUST perfectly match the provided schema structure.
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
    # --- THE FIX: PYTHON-DRIVEN PACING CALCULATION ---
    # 15s // 4 = 3 scenes. 30s // 4 = 7 scenes.
    exact_scenes = max(3, target_duration // 4)
    print(f"\n[AGENT: DIRECTOR] Drafting {exact_scenes}-scene structured storyboard for {target_duration}s target...")
    
    max_retries = 3
    attempt = 0
    final_storyboard = None
    
    while attempt < max_retries:
        try:
            draft_data = _generate_initial_draft(video_url, target_duration, exact_scenes, creative_directive, production_type)
            
            # The Critic now runs on every pass to guarantee pacing length, even if there is no creative directive
            print(f"[AGENT: CRITIC] Evaluating draft against User Directive and {exact_scenes}-scene constraint...")
            draft_data = _critique_and_revise(draft_data, creative_directive, target_duration, exact_scenes, video_url) 
            
            # Final Python-side enforcement: brutally slice the array if the LLM hallucinated extra scenes
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