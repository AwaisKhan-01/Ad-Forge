"""Generator — renders each scene into a video chunk with the RIGHT payload for
the chosen model and the narrative thread intact.

Payload adapters (registry `payload` key):
  multi_image  -> image_urls list (Seedance, Gemini Omni)
  endframe     -> image_url + optional end_image_url (Kling O3, Wan 2.7)
  flf          -> first_frame_url + last_frame_url, BOTH required (Veo 3.1)

Frame-role rules (hard-won in v1):
  - end frame is set ONLY from an explicit role, never positionally — a hard-cut
    scene passes the previous frame as an IDENTITY reference, and locking onto
    it makes the clip animate backwards.
  - motion graphics chains own-frame -> next-frame so cuts disappear on
    end-frame models; the final scene ends free.
"""
import os
import uuid
import concurrent.futures
from typing import Dict, List, Optional

import fal_client
import requests

from app.new_pipeline.model_registry import VIDEO_MODELS, clamp_duration, video_price
from app.new_pipeline.prompt_engine import build_video_prompt

CHUNK_DIR = "outputs/chunks"
SECONDS_PER_WORD = 0.5
SPEECH_PAD = 0.8


def plan_durations(scenes: List[Dict], total: int, model_key: str) -> List[float]:
    """Speech-aware proportional split, clamped later to the model's range."""
    if not scenes:
        return []
    weights, needs = [], []
    for s in scenes:
        words = len((s.get("dialogue") or "").split())
        need = words * SECONDS_PER_WORD + (SPEECH_PAD if words else 0)
        needs.append(need)
        weights.append(max(need, 3.0))
    share = float(total) / sum(weights)
    return [max(w * share, n) for w, n in zip(weights, needs)]


def _adapt_payload(model_key: str, prompt: str, seconds: float, scene: Dict,
                   aspect_ratio: str, resolution: str,
                   frame_urls: List[str], end_frame: Optional[str]) -> Dict:
    """Build the request EXACTLY as this endpoint's schema demands."""
    m = VIDEO_MODELS[model_key]
    duration = clamp_duration(model_key, seconds)
    payload_kind = m["payload"]

    if payload_kind == "multi_image":
        args = {"prompt": prompt, "duration": duration, "image_urls": frame_urls[:m["max_images"]]}
        if aspect_ratio in (m.get("aspect_ratios") or []):
            args["aspect_ratio"] = aspect_ratio
        if m.get("resolutions") and resolution in m["resolutions"]:
            args["resolution"] = resolution
        if model_key == "seedance-2.0-fast":
            args["generate_audio"] = True
        return args

    if payload_kind == "endframe":
        args = {"prompt": prompt, "duration": duration, "image_url": frame_urls[0]}
        if end_frame:
            args["end_image_url"] = end_frame
        if "generate_audio_default" in m:
            args["generate_audio"] = m["generate_audio_default"]
        if m.get("resolutions"):
            args["resolution"] = resolution if resolution in m["resolutions"] else m["resolutions"][0]
        return args

    if payload_kind == "flf":                      # Veo: both frames REQUIRED
        last = end_frame or frame_urls[0]          # final scene: hold on its own frame
        return {"prompt": prompt, "duration": duration,
                "first_frame_url": frame_urls[0], "last_frame_url": last,
                "resolution": resolution if resolution in (m.get("resolutions") or []) else "720p",
                "generate_audio": True}

    raise ValueError(f"Unknown payload kind: {payload_kind}")


def _download(url: str) -> str:
    os.makedirs(CHUNK_DIR, exist_ok=True)
    path = os.path.join(CHUNK_DIR, f"{uuid.uuid4().hex[:10]}.mp4")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _is_content_flag(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in ("content_policy", "content checker", "flagged",
                                "moderation", "nsfw", "no video url"))


def render_segment(idx: int, scene: Dict, campaign: Dict, seconds: float,
                   model_key: str, aspect_ratio: str, resolution: str,
                   text_mode: str, end_frame: Optional[str],
                   identity_frame: Optional[str] = "") -> Dict:
    """Render one chunk. NEVER raises — a failed chunk must not sink the run."""
    m = VIDEO_MODELS[model_key]
    mg = bool(campaign.get("is_motion_graphic"))
    own = scene.get("generated_image_url") or ""
    scene["_has_end_frame"] = bool(end_frame)

    frames = [own]
    if identity_frame and m["payload"] == "multi_image":
        frames.append(identity_frame)   # identity ref is safe ONLY for list models

    prompt = build_video_prompt(scene, campaign, model_key,
                                text_mode=text_mode, motion_graphic=mg)
    args = _adapt_payload(model_key, prompt, seconds, scene,
                          aspect_ratio, resolution, frames, end_frame)

    base = {"idx": idx, "scene_number": scene.get("scene_number", idx + 1),
            "duration": seconds, "endpoint": m["endpoint"], "args": args,
            "model_id": model_key}
    print(f"[RENDER] scene {base['scene_number']} start={frames[0][-24:]} "
          f"end={(end_frame or 'free')[-24:]} dur={args.get('duration')}")
    try:
        res = fal_client.subscribe(m["endpoint"], arguments=args)
        url = (res.get("video") or {}).get("url")
        if not url:
            raise RuntimeError("no video url (often a silent content rejection)")
        return {**base, "status": "ok", "path": _download(url), "error": None}
    except Exception as e:
        msg = str(e) or type(e).__name__
        if _is_content_flag(msg):       # one bland-prompt retry, then give up
            try:
                safe = dict(args)
                safe["prompt"] = ("A clean professional commercial shot, smooth natural "
                                  "motion, cinematic lighting, high production value.")
                res = fal_client.subscribe(m["endpoint"], arguments=safe)
                url = (res.get("video") or {}).get("url")
                if url:
                    return {**base, "args": safe, "status": "ok",
                            "path": _download(url), "error": None,
                            "note": "recovered via neutral prompt after content flag"}
            except Exception as e2:
                msg = f"content flagged; neutral retry failed too: {e2}"
        print(f"[RENDER] scene {base['scene_number']} FAILED: {msg}")
        return {**base, "status": "failed", "path": None, "error": msg}


def render_all(scenes: List[Dict], campaign: Dict, model_key: str, total_duration: int,
               aspect_ratio: str, resolution: str, text_mode: str = "add",
               duration_overrides: Optional[List[float]] = None,
               on_progress=None) -> List[Dict]:
    """All chunks in parallel. Continuity per model family:
    end-frame models chain own->next; list models get prev frame as identity ref
    on 'cut', prev-then-own ordering on 'continuous'."""
    durations = (duration_overrides if duration_overrides and
                 len(duration_overrides) == len(scenes)
                 else plan_durations(scenes, total_duration, model_key))
    m = VIDEO_MODELS[model_key]
    mg = bool(campaign.get("is_motion_graphic"))
    results = []

    def job(i, sc):
        nxt = scenes[i + 1].get("generated_image_url") if i + 1 < len(scenes) else None
        prev = scenes[i - 1].get("generated_image_url") if i > 0 else None
        end_frame, identity = None, None
        if m["payload"] in ("endframe", "flf"):
            if mg or sc.get("transition") == "continuous":
                end_frame = nxt            # land exactly where the next clip starts
        elif prev:
            identity = prev                # style/identity hint for list models
        return render_segment(i, sc, campaign, durations[i], model_key,
                              aspect_ratio, resolution, text_mode, end_frame, identity)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(job, i, sc) for i, sc in enumerate(scenes)]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            results.append(r)
            if on_progress:
                on_progress(r)
    results.sort(key=lambda r: r["idx"])
    return results


def rerender_segment(segments: List[Dict], scene_number, prompt_override: str = "",
                     image_override: str = "") -> List[Dict]:
    """Re-render exactly one chunk with user-corrected inputs."""
    segments = [dict(s) for s in segments]
    target = next((s for s in segments
                   if str(s.get("scene_number")) == str(scene_number)), None)
    if not target:
        raise ValueError(f"scene {scene_number} not found")
    args = dict(target["args"])
    if prompt_override.strip():
        args["prompt"] = prompt_override.strip()
    if image_override.strip():
        for k in ("image_url", "first_frame_url"):
            if k in args:
                args[k] = image_override.strip()
        if "image_urls" in args:
            args["image_urls"][0] = image_override.strip()
    res = fal_client.subscribe(target["endpoint"], arguments=args)
    url = (res.get("video") or {}).get("url")
    if not url:
        raise RuntimeError("re-render returned no video")
    target.update(status="ok", path=_download(url), error=None, args=args)
    return segments


def estimate_run_cost(scenes: List[Dict], model_key: str, total_duration: int,
                      resolution: str,
                      duration_overrides: Optional[List[float]] = None) -> float:
    durations = (duration_overrides if duration_overrides and
                 len(duration_overrides) == len(scenes)
                 else plan_durations(scenes, total_duration, model_key))
    return sum(video_price(model_key, d, resolution) for d in durations)
