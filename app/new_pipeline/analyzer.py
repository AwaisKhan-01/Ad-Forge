"""
Implements the 3-Pass Agency Architecture.
Features Semantic Timecoding and Aggressive Asset Injection.
Located in: app/new_pipeline/analyzer.py
"""
import re
import fal_client
from app.config import NEMOTRON_MODEL

SYSTEM_PROMPT = "You are a senior video advertising director and VFX supervisor. Follow instructions strictly."

def _run_pass_1_macro_strategy(video_url: str, target_duration: int) -> str:
    print(f"\n[AGENCY PASS 1] Analyzing reference ad structural psychology for a {target_duration}s target...")
    example_count = max(3, target_duration // 4) 
    
    prompt = f"""Watch this reference video. Extract the core marketing architecture. Do NOT list specific product assets.
    
CRITICAL TASK: Analyze the pacing (cut rate) of this video. Based on its rhythm, calculate the optimal number of distinct camera shots/scenes required for a {target_duration}-second commercial. 

Output strictly in this format:
RECOMMENDED SCENE COUNT: (Integer only. e.g., {example_count})
CORE HOOK: (Psychological hook in the first 3 seconds)
VALUE PROPOSITION: (Core problem and solution)
BRAND VIBE: (Aesthetic description, lighting, pacing)
CALL TO ACTION: (Final message style)
"""
    result = fal_client.subscribe(NEMOTRON_MODEL, arguments={"prompt": prompt, "system_prompt": SYSTEM_PROMPT, "video_url": video_url, "max_tokens": 1024, "temperature": 0.1, "reasoning_mode": "no_think"})
    return result.get("output", "").strip()


def _run_pass_2_copywriter(video_url: str, macro_strategy: str, target_duration: int, creative_directive: str, production_type: str) -> str:
    print(f"\n[AGENCY PASS 2] Writing adaptive script for: {production_type}...")
    max_words = int(target_duration * 2.5)
    directive_text = f"USER CREATIVE DIRECTIVE: {creative_directive if creative_directive else 'Direct Adaptation.'}"

    if "No Voiceover" in production_type:
        audio_rule = "1. NO SPOKEN VOICEOVER. Write short, punchy ON-SCREEN TEXT CARDS."
        format_instructions = "On-Screen Text: (Sequential text cards)\nMusic Track: (Style)\nTimeline: Split pacing evenly."
    elif "Motion Graphic" in production_type:
        audio_rule = "1. NO VOICEOVER. Focus purely on kinetic motion transitions and SFX."
        format_instructions = "Music & SFX Profile: (Sound design layout)\nTimeline: Split pacing across blocks."
    elif "On-Screen Actor" in production_type:
        audio_rule = f"1. ON-SCREEN DIALOGUE. Write an active spoken script (under {max_words} words)."
        format_instructions = f"Dialogue: (Script under {max_words} words)\nMusic: (Track style)\nTimeline: Split pacing evenly."
    else:
        audio_rule = f"1. OFF-SCREEN VOICEOVER. Write a continuous voiceover script under {max_words} words."
        format_instructions = f"Voiceover: (Script under {max_words} words)\nMusic: (Track style)\nTimeline: Split pacing evenly."

    prompt = f"""You are an expert ad copywriter. Take this macro strategy and condense it into a {target_duration}-second timeline.
MACRO STRATEGY:
{macro_strategy}
{directive_text}
CRITICAL FORMAT RULES:
{audio_rule}

Output strictly in this format:
{format_instructions}
"""
    result = fal_client.subscribe(NEMOTRON_MODEL, arguments={"prompt": prompt, "system_prompt": SYSTEM_PROMPT, "video_url": video_url, "max_tokens": 1536, "temperature": 0.2, "reasoning_mode": "no_think"})
    return result.get("output", "").strip()


def _run_pass_3_video_spotter(video_url: str, optimized_timeline: str, exact_scenes: int, creative_directive: str, production_type: str) -> str:
    print(f"\n[AGENCY PASS 3] Mapping user assets to exactly {exact_scenes} intelligently calculated scenes...")
    
    audio_split = optimized_timeline.split('Timeline:')
    master_audio = audio_split[0].strip() if len(audio_split) > 0 else optimized_timeline
    
    if "Motion Graphic" in production_type:
        audio_label = "SFX:"
    elif "On-Screen Actor" in production_type:
        audio_label = "Dialogue Focus:"
    else:
        audio_label = "Audio:"

    prompt = f"""Watch the reference video again. Generate a detailed storyboard matching this timeline.
You MUST output EXACTLY {exact_scenes} scene blocks.

CRITICAL ASSET REPLACEMENT PROTOCOL:
User Directive: "{creative_directive if creative_directive else 'No custom assets'}"
If the user's directive asks to replace a person, actor, product, or logo, YOU MUST explicitly type the exact string "#Image2" into the 'Visual:' section of EVERY scene where they appear. 
Example: "A close up of #Image2 talking to the camera..."
If you do not write "#Image2", the replacement will fail.

OUTPUT EXACTLY IN THIS FORMAT:

[MASTER AUDIO]
{master_audio}

SCENE 1
Timestamp: (Look at the ORIGINAL source video. Write the ACTUAL timecode where this visual occurs, e.g., 00:12 - 00:15. DO NOT compress this to match the target duration.)
Visual: (Dense physical description. MUST contain #Image2 if replacing an actor/product)
Motion: (Camera movement description)
{audio_label} (Audio cues for this scene)

SCENE 2
(Continue pattern up to EXACTLY scene {exact_scenes})
"""
    result = fal_client.subscribe(NEMOTRON_MODEL, arguments={"prompt": prompt, "system_prompt": SYSTEM_PROMPT, "video_url": video_url, "max_tokens": 4096, "temperature": 0.1, "reasoning_mode": "no_think"})
    return result.get("output", "").strip()


def analyze_and_extract_scenes(video_url: str, target_duration: int, creative_directive: str = "", production_type: str = "Narrative Ad (Voiceover focused)") -> str:
    macro_strategy = _run_pass_1_macro_strategy(video_url, target_duration)
    scene_match = re.search(r"RECOMMENDED SCENE COUNT:\s*(\d+)", macro_strategy, re.IGNORECASE)
    exact_scenes = int(scene_match.group(1)) if scene_match else max(2, target_duration // 5)
    print(f"[*] Computed Pacing: {exact_scenes} distinct scenes required.")
    
    optimized_timeline = _run_pass_2_copywriter(video_url, macro_strategy, target_duration, creative_directive, production_type)
    final_storyboard = _run_pass_3_video_spotter(video_url, optimized_timeline, exact_scenes, creative_directive, production_type)
    
    return f"=== STRATEGY ===\n{macro_strategy}\n\n=== STORYBOARD ===\n{final_storyboard}".replace("\\n", "\n").replace(r"\n", "\n")