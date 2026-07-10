"""
Stitches video segments containing native audio streams.
Located in: app/new_pipeline/stitcher.py
"""
import subprocess
import os
import uuid

OUTPUT_DIR = "outputs"

def stitch_segments_natively(segment_paths: list[str]) -> str:
    """
    Concatenates video segments that already contain their own audio streams.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    list_file = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_list.txt")
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_final.mp4")
    
    with open(list_file, "w") as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
            
    # Mux natively without an external audio track
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
        "-c:v", "libx264", "-c:a", "aac", "-y", output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    
    if os.path.exists(list_file):
        os.remove(list_file)
        
    return output_path