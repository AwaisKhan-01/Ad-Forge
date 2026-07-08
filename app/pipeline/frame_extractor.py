"""Extracts representative frames from a video for image-reference models."""
import subprocess
import os
import uuid

FRAMES_DIR = "frames"


def _get_duration(video_path: str) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


def extract_frames(video_path: str, count: int = 3) -> list[str]:
    """Extracts `count` evenly-spaced frames across the FULL video. Kept for backward use."""
    duration = _get_duration(video_path)
    return extract_frames_in_range(video_path, 0, duration, count)


def extract_frames_in_range(video_path: str, start_sec: float, end_sec: float, count: int) -> list[str]:
    """Extracts `count` evenly-spaced frames, but ONLY within [start_sec, end_sec] of the video."""
    os.makedirs(FRAMES_DIR, exist_ok=True)
    span = max(end_sec - start_sec, 0.1)

    paths = []
    for i in range(count):
        timestamp = start_sec + (span / (count + 1)) * (i + 1)
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
    """Used for segment-chaining continuity."""
    out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
    subprocess.run(
        ["ffmpeg", "-sseof", "-1", "-i", video_path,
         "-update", "1", "-q:v", "2", out_path, "-y"],
        capture_output=True,
    )
    return out_path