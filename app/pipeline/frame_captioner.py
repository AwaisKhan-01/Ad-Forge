"""Captions each extracted frame with a vision model so fusion is grounded in real image content."""
import fal_client


def caption_frame(image_url: str) -> str:
    """Returns a short factual caption of what's actually in this frame."""
    result = fal_client.subscribe(
        "fal-ai/any-llm/vision",
        arguments={
            "prompt": "Describe exactly what is visible: subject, pose, framing, background, lighting. Be factual, no interpretation.",
            "image_url": image_url,
        },
    )
    output = result.get("output")
    if not output:
        raise RuntimeError("Frame captioning returned no output.")
    return output.strip()


def caption_frames(image_urls: list[str]) -> list[str]:
    """Captions each frame in order, returns list matching @Image1..N order."""
    return [caption_frame(url) for url in image_urls]