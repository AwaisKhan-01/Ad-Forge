"""Extracts representative frames from a video for image-reference models."""
import subprocess
import os
import uuid

FRAMES_DIR = "frames"


def extract_frames(video_path: str, count: int = 3) -> list[str]:
    """Extracts `count` evenly-spaced frames, returns local file paths."""
    os.makedirs(FRAMES_DIR, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())

    paths = []
    for i in range(count):
        timestamp = (duration / (count + 1)) * (i + 1)
        out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
        subprocess.run(
            ["ffmpeg", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", "-q:v", "2", out_path, "-y"],
            capture_output=True,
        )
        if os.path.exists(out_path):
            paths.append(out_path)

    return paths


def extract_last_frame(video_path: str) -> str:
    """Used later for segment-chaining continuity."""
    out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
    subprocess.run(
        ["ffmpeg", "-sseof", "-1", "-i", video_path,
         "-update", "1", "-q:v", "2", out_path, "-y"],
        capture_output=True,
    )
    return out_path