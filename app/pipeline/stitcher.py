"""Concatenates generated video segments into one final ad using ffmpeg."""
import subprocess
import os
import uuid

OUTPUT_DIR = "outputs"


def stitch_segments(segment_paths: list[str]) -> str:
    """Concatenates segments in order, re-encoding for consistent codec/resolution."""
    if len(segment_paths) == 1:
        return segment_paths[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    list_file = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_list.txt")
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_final.mp4")

    with open(list_file, "w") as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c:v", "libx264", "-c:a", "aac", "-y", output_path],
        capture_output=True, check=True,
    )

    os.remove(list_file)
    return output_path