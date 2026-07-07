import fal_client
from app.config import SEEDANCE_MODEL
from app.pipeline.fusion import fuse_prompt


def generate_video(video_url: str, breakdown: str, requested_changes: str,
                    duration: str = "auto", resolution: str = "720p") -> str:
    """Fuses breakdown + changes into one brief, then calls Seedance reference-to-video."""
    final_prompt = fuse_prompt(breakdown, requested_changes)
    final_prompt = f"Recreate @Video1's structure. {final_prompt}"

    result = fal_client.subscribe(
        SEEDANCE_MODEL,
        arguments={
            "prompt": final_prompt,
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