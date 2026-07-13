"""
Stitches video segments containing native audio streams instantly via Stream Copy.
Located in: app/new_pipeline/stitcher.py
"""
import subprocess
import os
import uuid

OUTPUT_DIR = "outputs"

def stitch_segments_natively(segment_paths: list[str]) -> str:
    """
    Concatenates video segments that already contain their own audio streams.
    Utilizes -c copy to instantly mux files without quality loss or re-encoding.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    list_file = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_list.txt")
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_final.mp4")
    
    with open(list_file, "w") as f:
        for path in segment_paths:
            # Format required by FFmpeg concat demuxer
            f.write(f"file '{os.path.abspath(path)}'\n")
            
    # --- THE FIX: TRUE NATIVE MUXING ---
    # -c copy prevents re-encoding, preserving absolute quality and saving time
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", "-y", output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG ERROR] Failed to stitch segments. Details:\n{e.stderr}")
        raise RuntimeError("Video stitching failed.")
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)
        
    return output_path