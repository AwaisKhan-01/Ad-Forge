"""Downloads a video from an Instagram/TikTok URL using yt-dlp."""
import os
import uuid
import yt_dlp

DOWNLOAD_DIR = "downloads"


def download_video(url: str) -> str:
    """Downloads video from given URL, returns local file path."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())
    out_path = os.path.join(DOWNLOAD_DIR, f"{file_id}.mp4")

    ydl_opts = {
        "outtmpl": out_path,
        "format": "mp4/best",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(out_path):
        raise RuntimeError(f"Download failed for URL: {url}")

    return out_path
