"""Uploads a local video file to fal storage, returns a public URL Nemotron/Seedance can read."""
import fal_client


def upload_video(local_path: str) -> str:
    """Uploads file to fal storage, returns hosted URL."""
    url = fal_client.upload_file(local_path)
    if not url:
        raise RuntimeError(f"Upload failed for: {local_path}")
    return url
