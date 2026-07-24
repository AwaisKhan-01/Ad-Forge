"""Downloads a video from an Instagram/TikTok URL using yt-dlp, with a local cache."""
import os
import hashlib
import yt_dlp

DOWNLOAD_DIR = "downloads"


def download_video(url: str) -> str:
    """Return a local path for the video at `url`, downloading only if not cached.

    The filename is derived from a hash of the URL, so the same reference maps to
    the same file and is fetched once. Re-running with the same link is instant.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    # Deterministic name per URL -> natural cache key.
    file_id = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(DOWNLOAD_DIR, f"{file_id}.mp4")

    # Cache hit: reuse the already-downloaded reference.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[DOWNLOADER] Using cached reference: {out_path}")
        return out_path

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
