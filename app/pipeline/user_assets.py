"""Uploads user-supplied reference images and builds their @ImageN description text."""
from app.pipeline.uploader import upload_video


def process_user_images(image_paths: list[str], descriptions: str) -> tuple[list[str], str]:
    """Uploads images, returns (hosted_urls, prompt_note describing each by index)."""
    if not image_paths:
        return [], ""

    urls = [upload_video(p) for p in image_paths]
    note = f"User-provided reference images and how to use them: {descriptions}" if descriptions else ""
    return urls, note