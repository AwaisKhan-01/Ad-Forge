"""Uploads user-supplied reference images. Descriptions now live directly in the user's prompt as @ImageN."""
from app.pipeline.uploader import upload_video


def process_user_images(image_paths: list[str]) -> list[str]:
    """Uploads images in order, returns hosted URLs matching @Image1, @Image2... order."""
    if not image_paths:
        return []
    return [upload_video(p) for p in image_paths]