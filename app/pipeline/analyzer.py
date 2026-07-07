"""Sends the uploaded ad video to Nemotron and gets back a structured creative breakdown."""
import fal_client
from app.config import NEMOTRON_MODEL

SYSTEM_PROMPT = """You are an expert ad creative analyst. Watch the video and extract, in detail:
1. Shot list: number of shots, order, duration of each
2. Camera work: angles, movement (pan/zoom/static), framing
3. Lighting and color grade/mood
4. Subject/product: what's shown, how it's positioned, actions taken
5. On-screen text: exact wording, placement, timing, font style if visible
6. Pacing and cuts: fast/slow, transition style
7. Audio cues if inferable: music mood, voiceover tone, sound effects

Output as clear labeled sections. Be precise and concrete, not generic."""


def analyze_video(video_url: str, user_notes: str = "") -> str:
    """Calls Nemotron on the uploaded video, returns the raw text breakdown."""
    prompt = "Analyze this ad video in full detail."
    if user_notes:
        prompt += f" Additional focus requested: {user_notes}"

    result = fal_client.subscribe(
        NEMOTRON_MODEL,
        arguments={
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "video_url": video_url,
            "reasoning_mode": "think",
        },
    )

    output = result.get("output")
    if not output:
        raise RuntimeError("Nemotron returned no output.")

    return output