"""Combines the creative breakdown + user-requested changes into a Seedance prompt and generates the video."""
import fal_client
from app.config import SEEDANCE_MODEL


def build_prompt(breakdown: str, requested_changes: str) -> str:
    """Merges Nemotron's breakdown with the user's desired edits into a Seedance instruction prompt."""
    return (
        f"Recreate the composition, pacing, and shot structure of @Video1.\n\n"
        f"Reference breakdown:\n{breakdown}\n\n"
        f"Apply these changes:\n{requested_changes}"
    )


def generate_video(video_url: str, breakdown: str, requested_changes: str,
                    duration: int = 5, resolution: str = "720p") -> str:
    """Calls Seedance reference-to-video, returns the generated video URL."""
    prompt = build_prompt(breakdown, requested_changes)

    result = fal_client.subscribe(
        SEEDANCE_MODEL,
        arguments={
            "prompt": prompt,
            "video_urls": [video_url],
            "duration": duration,
            "resolution": resolution,
        },
    )

    video = result.get("video", {})
    output_url = video.get("url")
    if not output_url:
        raise RuntimeError("Seedance returned no video output.")

    return output_url