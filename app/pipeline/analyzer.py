"""Sends the ad video to Nemotron; breakdown style depends on which generation model will consume it."""
import fal_client
from app.config import NEMOTRON_MODEL

SYSTEM_PROMPT_SEEDANCE = """You are an ad creative analyst. Watch the video and extract:
1. Shot list: number of shots, order, duration of each
2. Camera work: angles, movement, framing
3. Lighting and color grade/mood
4. Subject/product: what's shown, positioning, actions
5. On-screen text: exact wording, placement, timing
6. Pacing and cuts
7. Audio cues if inferable

Output labeled sections. Be precise and concrete, not generic."""

SYSTEM_PROMPT_OMNI = """You are an ad creative analyst preparing notes for a keyframe-based image-to-video generator.
Break the video into {n} distinct key moments spaced evenly across its runtime. For EACH moment describe:
- Composition: subject position, framing, what's in view
- Action/pose at that exact moment
- Lighting/color at that moment
- How it visually transitions into the next moment

Output as "Moment 1:", "Moment 2:", etc. Be precise and concrete, not generic."""


def analyze_video(video_url: str, model_id: str, user_notes: str = "", frame_count: int = 5) -> str:
    """Calls Nemotron on the video. Breakdown format adapts to the target generation model."""
    if model_id == "gemini-omni-flash":
        system_prompt = SYSTEM_PROMPT_OMNI.format(n=frame_count)
        prompt = f"Analyze this ad video and break it into {frame_count} key moments."
    else:
        system_prompt = SYSTEM_PROMPT_SEEDANCE
        prompt = "Analyze this ad video in full detail."

    if user_notes:
        prompt += f" Additional focus requested: {user_notes}"

    result = fal_client.subscribe(
        NEMOTRON_MODEL,
        arguments={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "video_url": video_url,
            "reasoning_mode": "no_think",
        },
    )

    output = result.get("output")
    if not output:
        raise RuntimeError("Nemotron returned no output.")

    if "<think>" in output and "</think>" in output:
        output = output.split("</think>", 1)[1].strip()

    return output