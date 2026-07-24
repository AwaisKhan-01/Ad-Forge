"""
Stitches video segments into a cohesive ad.

Two paths:
  - stitch_segments_natively(): smooth crossfade transitions (xfade/acrossfade)
    so the cuts feel like intentional edits instead of jarring hard splices.
  - _concat_copy(): the original lossless hard-cut concat, used as a fallback
    when the crossfade filtergraph can't run (e.g. mismatched formats).

Located in: app/new_pipeline/stitcher.py
"""
import os
import uuid
import subprocess

OUTPUT_DIR = "outputs"

# How long each cross-scene transition lasts. Kept very short: a long dissolve
# reads as mushy/amateur in a fast-paced ad, AND every crossfade eats this many
# seconds of audio at the boundary — which clips the start/end of dialogue.
XFADE_DURATION = 0.15


def _probe_duration(path: str) -> float:
    """Return a segment's real duration in seconds (models rarely return the
    exact requested length, so we must measure, not assume)."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            text=True, timeout=30,
        ).strip()
        return float(out)
    except Exception:
        return 0.0


def _concat_copy(segment_paths: list[str]) -> str:
    """Original behavior: instant lossless hard-cut concat via stream copy."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    list_file = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_list.txt")
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_final.mp4")

    with open(list_file, "w") as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", "-y", output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG ERROR] concat failed:\n{e.stderr}")
        raise RuntimeError("Video stitching failed.")
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)
    return output_path


def stitch_segments_natively(segment_paths: list[str], hard_cut: bool = False) -> str:
    """Concatenate segments with short crossfade transitions between each cut.

    Builds a chained xfade (video) + acrossfade (audio) filtergraph. Each
    transition overlaps the outgoing and incoming clip by XFADE_DURATION, so the
    running offset for cut N is (sum of prior durations) - N*XFADE_DURATION.

    Falls back to a plain hard-cut concat if there is only one clip, or if the
    crossfade render fails for any reason — we never want to lose a finished ad.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    segment_paths = [p for p in segment_paths if p and os.path.exists(p)]
    # Motion graphics: clips deliberately end mid-exit and open mid-arrival, so a
    # dissolve would blend two DIFFERENT movements into mush. Hard cut instead.
    if hard_cut and len(segment_paths) > 1:
        return _concat_copy(segment_paths)
    if not segment_paths:
        raise RuntimeError("No video segments to stitch.")
    if len(segment_paths) == 1:
        # Nothing to transition between; just hand back the single clip re-muxed.
        return _concat_copy(segment_paths)

    durations = [_probe_duration(p) for p in segment_paths]
    if any(d <= 0 for d in durations):
        print("[STITCHER] Could not probe a segment duration; using hard-cut concat.")
        return _concat_copy(segment_paths)

    # A transition can't be longer than the shorter of the two clips it joins.
    xfade = min(XFADE_DURATION, min(durations) / 2)
    if xfade <= 0.05:
        return _concat_copy(segment_paths)

    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_final.mp4")

    cmd = ["ffmpeg"]
    for p in segment_paths:
        cmd += ["-i", p]

    # Normalize each input's video (consistent fps/format keeps xfade stable
    # across clips that may differ slightly) and label v/a streams.
    filters = []
    for i in range(len(segment_paths)):
        filters.append(f"[{i}:v]settb=AVTB,fps=30,format=yuv420p[v{i}]")
        filters.append(f"[{i}:a]aresample=async=1[a{i}]")

    # Chain the crossfades. offset = cumulative visible time before this cut.
    v_prev, a_prev = "v0", "a0"
    offset = durations[0] - xfade
    for i in range(1, len(segment_paths)):
        v_out = f"vx{i}"
        a_out = f"ax{i}"
        filters.append(
            f"[{v_prev}][v{i}]xfade=transition=fade:duration={xfade:.3f}:offset={offset:.3f}[{v_out}]"
        )
        filters.append(
            f"[{a_prev}][a{i}]acrossfade=d={xfade:.3f}[{a_out}]"
        )
        v_prev, a_prev = v_out, a_out
        # Each crossfade shortens the timeline by one xfade overlap.
        offset += durations[i] - xfade

    filter_complex = ";".join(filters)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", f"[{v_prev}]", "-map", f"[{a_prev}]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-y", output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        # Crossfade is best-effort; never let it cost the user a finished render.
        print(f"[STITCHER] Crossfade render failed, falling back to hard cut.\n{e.stderr[-800:]}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return _concat_copy(segment_paths)
