"""Lightweight on-disk session persistence.

gr.State lives only in the browser tab and is lost the moment the server
restarts (e.g. after a code change) — which would throw away an expensive
storyboard of many generated frames. This module autosaves the important state
to a JSON file on disk so it can be restored after a relaunch without
regenerating (and re-paying for) anything.

Only JSON-serializable pipeline state is stored: the storyboard frames (which
already hold uploaded fal image URLs), the rendered-segment state, and a few
lightweight settings. No binary/media is copied — the URLs point at fal.
"""
import os
import json
import time

SESSION_DIR = "sessions"
_LATEST = os.path.join(SESSION_DIR, "latest_session.json")


def _safe(obj):
    """Drop anything that isn't JSON-serializable rather than fail the save."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return None


def save_session(storyboard_frames=None, video_segments=None, extra=None) -> None:
    """Persist the current session to disk. Never raises — autosave must not
    interrupt the user's work if the disk write fails.

    Fields left as None are MERGED from the existing saved session, so saving
    only the segments (e.g. after a re-render) never wipes the stored storyboard.
    """
    try:
        os.makedirs(SESSION_DIR, exist_ok=True)
        prev = load_session()
        payload = {
            "saved_at": time.time(),
            "storyboard_frames": (_safe(storyboard_frames) if storyboard_frames is not None
                                  else prev.get("storyboard_frames")) or [],
            "video_segments": (_safe(video_segments) if video_segments is not None
                               else prev.get("video_segments")) or [],
            "extra": (_safe(extra) if extra is not None else prev.get("extra")) or {},
        }
        tmp = _LATEST + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, _LATEST)  # atomic swap so a crash never leaves a half file
    except Exception as e:
        print(f"[SESSION] Autosave skipped: {e}")


def load_session() -> dict:
    """Return the last saved session, or an empty structure if none exists."""
    try:
        if os.path.exists(_LATEST):
            with open(_LATEST) as f:
                return json.load(f)
    except Exception as e:
        print(f"[SESSION] Could not load previous session: {e}")
    return {"storyboard_frames": [], "video_segments": [], "extra": {}}


def has_session() -> bool:
    data = load_session()
    return bool(data.get("storyboard_frames"))


def session_age_str() -> str:
    """Human-readable age of the saved session, for the restore prompt."""
    data = load_session()
    ts = data.get("saved_at")
    if not ts:
        return ""
    mins = max(0, int((time.time() - ts) / 60))
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    hours = mins // 60
    return f"{hours}h {mins % 60}m ago"
