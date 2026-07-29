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
    # Audio is split so spoken words are never confused with background score.
    # Only `dialogue` is ever spoken by the subject or rendered as typography.
    dialogue: str = Field(default="", description="ONLY the literal words spoken aloud in this scene. Empty string if nobody speaks. NEVER describe music or sound effects here.")
    on_screen_text: str = Field(default="", description="Short punchy copy to RENDER as animated typography (motion graphics). Max ~6 words. Not speech.")
    transition: str = Field(default="cut", description="How this scene follows the PREVIOUS one in the reference: 'continuous' (same unbroken shot, camera kept rolling) or 'cut' (a new shot). First scene is always 'cut'.")
    sfx_music: str = Field(default="", description="Background music / sound-effect DESCRIPTION only (e.g. 'upbeat synth'). Never spoken aloud, never shown as text.")
    audio: str = Field(default="", description="DEPRECATED legacy combined field. Prefer dialogue + sfx_music.")
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

def is_motion_graphic(production_type: str) -> bool:
    """True when the user asked for a designed/kinetic-graphics ad rather than
    live-action narrative. Every stage tunes its prompts off this flag."""
    return "motion graphic" in (production_type or "").lower()


# Extra rules injected only for motion-graphics runs. Live-action rules (dialogue,
# lip-sync, camera work) are actively harmful here, so they are replaced.
_MOTION_GRAPHIC_RULES = """
MOTION-GRAPHICS MODE (OVERRIDES CONFLICTING RULES ABOVE):
M1. NO SPOKEN DIALOGUE. Nobody talks on screen. 'dialogue' MUST be "" for every scene.
M2. ON-SCREEN COPY: put the short punchy words that should APPEAR as animated
    typography in 'on_screen_text' (max 6 words per scene, ALL CAPS-friendly, no
    full sentences). This is design copy, not speech.
M3. VISUAL = DESIGNED COMPOSITION. Describe the layout and elements: background
    treatment, product/logo placement, negative space, where the type sits.
    CRITICAL: describe the REFERENCE VIDEO'S ACTUAL rendering style — if it is
    dark 3D with glass/glow, say so; if it is photoreal, say so. Do NOT default to
    "flat vector" or "illustration" unless the reference genuinely looks that way.
M4. 'motion' = the scene's ANIMATION SCRIPT — the kinetic behaviour of the
    graphics, never a camera move. WATCH the shot and TRANSCRIBE its actual
    choreography as 3-6 timed beats covering the whole clip, e.g.:
    "0-20%: logo scales in with overshoot; 20-60%: headline staggers in word by
    word; 60-100%: panels wipe left revealing the end composition".
    Use motion-design vocabulary (slide in, mask reveal, scale punch, wipe,
    kinetic type stagger, morph, parallax drift). Every beat must describe what
    the REFERENCE actually does in that shot — never invent generic motion. A
    one-line summary is NOT acceptable; the video model animates exactly and
    only what you write here.
M5. VISUAL CONSISTENCY: all scenes share ONE design system. Repeat the same
    palette, type treatment and layout logic in every scene so the ad reads as a
    single designed piece, not unrelated frames.
M6. PACE: motion-graphic ads cut fast. Favour short, punchy scenes.
"""

def _generate_initial_draft(video_url: str, target_duration: int, exact_scenes: int, creative_directive: str, production_type: str, asset_instructions: str) -> dict:
    max_words = int(target_duration * 2.5)
    mg = is_motion_graphic(production_type)

    prompt = f"""Analyze the ENTIRE reference video and generate a highly condensed storyboard for a {target_duration}-second {production_type} commercial.

CRITICAL USER DIRECTIVE: "{creative_directive if creative_directive else 'Direct Adaptation'}"

USER ASSET INSTRUCTIONS:
{asset_instructions if asset_instructions else 'No custom production assets uploaded.'}

RULES:
1. NARRATIVE COMPRESSION: Compress the storyline into exactly {exact_scenes} scenes. Select moments from BEGINNING, MIDDLE, and END.
2. NATIVE AUDIO CONTINUITY: Listen to the speech, spoken dialogue, and voiceover natively present in the video file. Edit and condense the actual spoken dialogue into a seamless, continuous script (maximum {max_words} words total).
2b. SPLIT AUDIO (CRITICAL): You MUST separate audio into two different fields.
   - 'dialogue': ONLY the literal words a person says out loud, exactly as they should be spoken. If nobody speaks in this scene, use an empty string "".
   - 'sfx_music': a DESCRIPTION of background music or sound effects (e.g. "upbeat synth, no vocals").
   NEVER put music or sound descriptions in 'dialogue' — text in 'dialogue' gets spoken aloud by the on-screen person and rendered as on-screen typography, so putting "upbeat electronic music" there would make the actor literally say those words. Likewise never put spoken words in 'sfx_music'.
2c. DIALOGUE LENGTH PER SCENE: a scene can only hold about 2 words per second of its timestamp span. Keep each scene's 'dialogue' short enough to be spoken naturally within that scene's duration. Split long lines across scenes rather than overloading one.
3. MANDATORY ASSET MAPPING (CRITICAL): Read the 'USER ASSET INSTRUCTIONS' above, which lists the available assets as "Asset 1", "Asset 2", etc. Asset numbering starts at 1. You MUST replace the original subjects or objects in the video with the requested assets. If the user says "Asset 1 is the main person", you MUST write "#Asset1" instead of "man" or "woman" in the 'visual_prompt', AND you MUST put the integer 1 in the 'assigned_assets' array (e.g., [1]) for EVERY scene that character appears in. The number in 'assigned_assets' MUST EXACTLY MATCH the number in the "#Asset" tag you wrote in 'visual_prompt'. NEVER use 0 or any number greater than the number of assets listed. Do NOT leave the array empty if the subject is present.
4. FREEZE TIME (CRITICAL): The 'visual_prompt' MUST describe a STILL PHOTOGRAPH. NO TIMELINES. NO SEQUENCES. You are STRICTLY FORBIDDEN from using the word "then". Put all camera movement in the 'motion' field.
5. ESCAPING: NEVER use unescaped double quotes inside values. Use single quotes.
6. EXPLICIT KEYFRAME TARGETING: You must provide a 'keyframe_time' (e.g. '00:05'). This must be the EXACT second within the timestamp block that perfectly visually matches your static 'visual_prompt'.
7. SHOT CONTINUITY: For each scene set 'transition' based on what the REFERENCE video does at its start. Use 'continuous' if this scene is the SAME unbroken shot as the previous one (no cut — the camera kept rolling on the same subject/framing, e.g. a long single take). Use 'cut' if the reference cuts to a NEW shot (different angle, framing or moment). The FIRST scene is always 'cut'. Be accurate — do not mark everything 'cut' by default; long single-take ads should be mostly 'continuous'.
7. KEYFRAME SUBJECT MATCH (CRITICAL): The frame at 'keyframe_time' will be extracted from the source video and used as the composition base for image generation. It MUST actually contain the primary subject you describe in 'visual_prompt'. If your visual_prompt describes a PERSON (e.g. the on-screen actor speaking), 'keyframe_time' MUST point at a second where that PERSON is clearly on screen — NOT a second showing a UI screen, product shot, logo, or B-roll. If the scene is about the talking person, pick a timestamp block AND keyframe_time where the person's face/body is visible. Do NOT set keyframe_time to a moment showing a screen/graphic when the visual_prompt is about a person, or vice versa — the base frame and the described subject must be the SAME type of shot.

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
      "dialogue": "string",
      "on_screen_text": "string",
      "transition": "cut | continuous",
      "sfx_music": "string",
      "assigned_assets": []
    }}
  ]
}}
{_MOTION_GRAPHIC_RULES if mg else ""}
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
2. ENFORCE ASSET MAPPING: Asset numbering starts at 1 and must never exceed the number of assets listed in USER ASSET INSTRUCTIONS. Did the draft assign assets to the correct scenes? If the draft left 'assigned_assets' empty `[]` but the subject is clearly in the scene, YOU MUST FIX IT by adding the integer to the array (e.g., `[1]`) and injecting `#Asset1` into the visual_prompt. Verify that for EVERY scene the number inside each "#AssetN" tag in 'visual_prompt' EXACTLY MATCHES an entry in that scene's 'assigned_assets' array — if they disagree, correct the array to match the tags. Replace any `0` index with the correct 1-based number.
3. ENFORCE TIME FREEZE: Read every 'visual_prompt'. If you see the word "then" or any sequential action (e.g., "walks in, then sits"), DELETE IT. Rewrite it to describe a single, frozen moment.
3b. ENFORCE AUDIO SPLIT: 'dialogue' must contain ONLY literal spoken words. If a scene's 'dialogue' contains a music or sound description (e.g. "upbeat electronic music", "soft piano", "whoosh sfx"), MOVE that text into 'sfx_music' and set 'dialogue' to "" (or to the real spoken words if any exist). This is critical: 'dialogue' is spoken aloud by the on-screen person and shown as on-screen text.
4. KEYFRAME INTEGRITY: Ensure 'keyframe_time' is present and accurately reflects a static moment within the scene's 'timestamp'.
5. KEYFRAME SUBJECT MATCH (CRITICAL): For each scene, verify the frame at 'keyframe_time' actually shows the SAME subject type as the 'visual_prompt'. If visual_prompt describes a PERSON but keyframe_time points at a screen/UI/product/logo/B-roll moment (or vice versa), FIX keyframe_time (and the 'timestamp' block if needed) to a second where the described subject is genuinely on screen. The extracted base frame must match the described subject.

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
        # Emit dialogue and music on SEPARATE lines so the parser never routes a
        # music description into spoken_dialogue (which would make the on-screen
        # actor literally say "upbeat electronic music" and render it as text).
        # Falls back to the legacy combined 'audio' field for older responses.
        spoken = (scene.dialogue or "").strip()
        music = (scene.sfx_music or "").strip()
        if not spoken and not music and (scene.audio or "").strip():
            spoken = scene.audio.strip()
        output_str += (
            f"SCENE {scene.scene_number}\n"
            f"Timestamp: {scene.timestamp}\n"
            f"Keyframe: {scene.keyframe_time}\n"
            f"Visual: {scene.visual_prompt}\n"
            f"Motion: {scene.motion}\n"
            f"Spoken Dialogue: {spoken}\n"
            f"On-Screen Text: {(scene.on_screen_text or '').strip()}\n"
            f"Transition: {(scene.transition or 'cut').strip()}\n"
            f"SFX/Music: {music}\n"
            f"Assigned Assets: {json.dumps(scene.assigned_assets)}\n\n"
        )
    return output_str.strip()