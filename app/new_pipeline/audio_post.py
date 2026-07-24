"""Audio post: lay ONE continuous music bed under the finished ad, ducked under
speech.

The per-segment clips each carry their own audio (model VO/sfx). Stitching them
gives a coherent voice track but no consistent musical spine. This stage adds a
single music track across the whole timeline and, when ducking is on, uses
sidechain compression so the music automatically dips whenever the video's own
audio (dialogue) is present, then swells back in the gaps.

It is a pure post-step on the already-stitched file and is FAIL-OPEN: if no music
is supplied, or anything goes wrong, it returns the original video untouched. It
never costs the user a finished render.
"""
import os
import uuid
import shutil
import subprocess

OUTPUT_DIR = "outputs"

# How far the music ducks under speech. sidechaincompress does the work; these
# are conservative defaults that keep music audible but well under the VO.
_MUSIC_GAIN = 0.35          # baseline music level relative to full scale
_DUCK_THRESHOLD = 0.03      # voice level that starts ducking
_DUCK_RATIO = 8             # how hard to duck once triggered


def _has_audio_stream(path: str) -> bool:
    """True if the file has at least one audio stream (voice to duck under)."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            text=True, timeout=30,
        ).strip()
        return bool(out)
    except Exception:
        return False


def add_music_bed(video_path: str, music_path: str, duck_under_voice: bool = True) -> str:
    """Return a new video with `music_path` mixed under it, or `video_path`
    unchanged if music is absent / on any failure.

    Music is looped/trimmed to the video length (`-shortest`). When
    duck_under_voice is on and the video has a voice track, the music is
    sidechain-compressed by that voice so dialogue always stays on top.
    """
    if not music_path or not os.path.exists(music_path):
        return video_path
    if not os.path.exists(video_path):
        return video_path

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}_scored.mp4")

    has_voice = _has_audio_stream(video_path)

    # Input 0 = video (with its voice track), input 1 = music (looped).
    cmd = ["ffmpeg", "-i", video_path, "-stream_loop", "-1", "-i", music_path]

    if has_voice and duck_under_voice:
        # Duck the music under the video's own audio, then mix the two.
        filter_complex = (
            f"[1:a]volume={_MUSIC_GAIN}[music];"
            f"[0:a]asplit=2[voice_mix][voice_key];"
            f"[music][voice_key]sidechaincompress="
            f"threshold={_DUCK_THRESHOLD}:ratio={_DUCK_RATIO}:attack=20:release=300[ducked];"
            f"[voice_mix][ducked]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
    elif has_voice:
        # No ducking: simple mix of voice + music.
        filter_complex = (
            f"[1:a]volume={_MUSIC_GAIN}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
    else:
        # No voice track at all: music becomes the whole soundtrack.
        filter_complex = f"[1:a]volume={_MUSIC_GAIN}[aout]"

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", "-y", output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[AUDIO POST] Music bed failed, keeping original audio.\n{e.stderr[-800:]}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return video_path
