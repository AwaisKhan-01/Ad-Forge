"""AdForge v2 API — FastAPI backend.

Flow: reference -> blueprint (analyze) -> storyboard (job) -> render (job).
Long stages run as background jobs; the UI follows them via SSE. All state for
the active session is persisted to disk so a server restart loses nothing.
"""
import os
import shutil
import uuid
from typing import List, Optional

import fal_client
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.new_pipeline import jobs, session_store
from app.new_pipeline.analyzer import analyze_reference, scene_to_dict
from app.new_pipeline.downloader import download_video
from app.new_pipeline.frame_designer import design_frame, design_storyboard, text_mode_for
from app.new_pipeline.generator import (estimate_run_cost, plan_durations,
                                        render_all, rerender_segment)
from app.new_pipeline.model_registry import (DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL,
                                             IMAGE_MODELS, VIDEO_MODELS, clamp_duration,
                                             image_price)
from app.new_pipeline.stitcher import stitch_segments_natively
from app.new_pipeline.uploader import upload_video

app = FastAPI(title="AdForge Studio", version="2.0")
UPLOAD_DIR = "uploads"

# ------------------------------------------------------------------ state
# IMPORTANT: the live session starts EMPTY on every boot. It must never be
# eagerly pre-filled from the on-disk autosave — a previous project's assets/
# reference/blueprint would then be silently used by /api/analyze etc. even
# though the UI shows a blank canvas (restoring is an explicit user action,
# not something that happens just because the server started).
STATE: dict = {"scenes": [], "segments": [], "assets": []}


def _persist():
    session_store.save_session(
        storyboard_frames=STATE.get("scenes"),
        video_segments=STATE.get("segments"),
        extra={k: v for k, v in STATE.items() if k not in ("scenes", "segments")},
    )


def _load_snapshot() -> dict:
    """Read the on-disk autosave fresh — used ONLY by the explicit restore
    endpoint, never merged into the live STATE automatically."""
    saved = session_store.load_session()
    snap = dict(saved.get("extra") or {})
    snap["scenes"] = saved.get("storyboard_frames") or []
    snap["segments"] = saved.get("video_segments") or []
    snap.setdefault("assets", [])
    return snap


def _state_payload() -> dict:
    ref = STATE.get("reference_path")
    return {"reference": bool(ref),
            "reference_preview": f"/media/{os.path.basename(ref)}" if ref else None,
            "assets": len(STATE.get("assets", [])),
            "asset_previews": [f"/media/{os.path.basename(p)}"
                               for p in STATE.get("assets", [])],
            "campaign": STATE.get("campaign"),
            "scenes": STATE.get("scenes", []),
            "has_segments": bool(STATE.get("segments")),
            "final": (f"/media/{os.path.basename(STATE['final_video'])}"
                      if STATE.get("final_video") else None)}


# ------------------------------------------------------------------ schemas
class RefRequest(BaseModel):
    url: str


class AnalyzeRequest(BaseModel):
    api_key: str
    directive: str = ""
    duration: int = 15
    production_type: str = "Narrative Ad (On-Screen Actor / Dialogue)"


class StoryboardRequest(BaseModel):
    api_key: str
    aspect_ratio: str = "16:9"      # only 9:16 / 16:9 offered app-wide
    font_style: str = "chunky_display"
    image_model: str = DEFAULT_IMAGE_MODEL
    image_quality: str = "high"     # GPT Image 2 only
    resolution: str = ""            # Nano Banana only: 0.5K/1K/2K/4K


class FrameEdit(BaseModel):
    api_key: str
    scene_number: int
    prompt_override: str = ""
    image_urls_override: Optional[List[str]] = None


class RenderRequest(BaseModel):
    api_key: str
    video_model: str = DEFAULT_VIDEO_MODEL
    resolution: str = "720p"
    aspect_ratio: str = "16:9"
    duration_overrides: Optional[List[float]] = None
    max_budget: float = 2.0


class SegmentEdit(BaseModel):
    api_key: str
    scene_number: int
    prompt_override: str = ""
    image_override: str = ""


def _set_key(key: str):
    if not key:
        raise HTTPException(400, "FAL API key required")
    os.environ["FAL_KEY"] = key
    fal_client.api_key = key


# ------------------------------------------------------------------ reference
def _reset_downstream():
    """A new reference starts a new project: stale blueprint/storyboard/render
    state must not leak into it (assets are kept — they were just uploaded)."""
    for k in ("campaign", "scenes", "segments", "final_video", "target_duration"):
        STATE.pop(k, None)
    STATE["scenes"], STATE["segments"] = [], []


@app.post("/api/reference/url")
def reference_from_url(req: RefRequest):
    path = download_video(req.url)
    STATE["reference_path"] = path
    _reset_downstream()
    _persist()
    return {"path": path, "preview": f"/media/{os.path.basename(path)}"}


@app.post("/api/reference/upload")
async def reference_upload(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{file.filename}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    STATE["reference_path"] = path
    _reset_downstream()
    _persist()
    return {"path": path, "preview": f"/media/{os.path.basename(path)}"}


@app.post("/api/assets/upload")
async def asset_upload(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{file.filename}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    assets = STATE.setdefault("assets", [])
    assets.append(path)
    _persist()
    return {"index": len(assets), "tag": f"Asset {len(assets)}",
            "preview": f"/media/{os.path.basename(path)}"}


@app.delete("/api/assets")
def assets_clear():
    STATE["assets"] = []
    _persist()
    return {"ok": True}


# ------------------------------------------------------------------ blueprint
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    _set_key(req.api_key)
    ref = STATE.get("reference_path")
    if not ref or not os.path.exists(ref):
        raise HTTPException(400, "Load a reference video first")
    assets = STATE.get("assets", [])
    manifest = "; ".join(f"Asset {i + 1}: {os.path.basename(p)}"
                         for i, p in enumerate(assets)) or "none"
    ref_url = upload_video(ref)
    campaign = analyze_reference(ref_url, req.directive, req.duration,
                                 req.production_type, manifest)
    STATE["campaign"] = campaign.model_dump(exclude={"scenes"})
    STATE["scenes"] = [scene_to_dict(s, campaign) for s in campaign.scenes]
    STATE["target_duration"] = req.duration
    STATE["production_type"] = req.production_type
    _persist()
    return {"campaign": STATE["campaign"], "scenes": STATE["scenes"]}


@app.put("/api/scenes")
def update_scenes(scenes: List[dict]):
    """The blueprint is fully editable before any money is spent."""
    STATE["scenes"] = scenes
    _persist()
    return {"ok": True, "count": len(scenes)}


# ------------------------------------------------------------------ storyboard
@app.post("/api/storyboard")
def storyboard(req: StoryboardRequest):
    _set_key(req.api_key)
    scenes, ref = STATE.get("scenes"), STATE.get("reference_path")
    if not scenes:
        raise HTTPException(400, "Analyze the reference first")
    campaign = STATE.get("campaign", {})
    STATE.update(font_style=req.font_style, image_model=req.image_model,
                 image_quality=req.image_quality,
                 resolution=req.resolution, aspect_ratio=req.aspect_ratio)

    def run(job):
        def progress(scene):
            job.emit("frame", scene_number=scene.get("scene_number"),
                     url=scene.get("generated_image_url"), error=scene.get("image_error"),
                     warning=scene.get("prompt_warning"))
        done = design_storyboard(scenes, campaign, STATE.get("assets", []), ref,
                                 req.aspect_ratio, req.font_style, req.image_model,
                                 req.image_quality, req.resolution,
                                 on_progress=progress)
        STATE["scenes"] = done
        _persist()
        return {"scenes": done,
                "image_cost": len(done) * image_price(req.image_model, req.aspect_ratio,
                                                      req.image_quality)}
    return {"job_id": jobs.start("storyboard", run).id}


@app.post("/api/storyboard/frame")
def regen_frame(req: FrameEdit):
    _set_key(req.api_key)
    scenes = STATE.get("scenes") or []
    scene = next((s for s in scenes
                  if int(s.get("scene_number", -1)) == req.scene_number), None)
    if not scene:
        raise HTTPException(404, f"scene {req.scene_number} not found")
    updated = design_frame(scene, STATE.get("campaign", {}),
                           [a if str(a).startswith("http") else upload_video(a)
                            for a in STATE.get("assets", [])],
                           STATE.get("reference_path"), STATE.get("aspect_ratio", "16:9"),
                           STATE.get("font_style", "chunky_display"),
                           STATE.get("image_model", DEFAULT_IMAGE_MODEL),
                           STATE.get("image_quality", "high"),
                           STATE.get("resolution", ""),
                           prompt_override=req.prompt_override,
                           images_override=req.image_urls_override)
    _persist()
    return updated


# ------------------------------------------------------------------ render
@app.get("/api/timeline")
def timeline(video_model: str = DEFAULT_VIDEO_MODEL):
    scenes = STATE.get("scenes") or []
    if not scenes:
        raise HTTPException(400, "No scenes yet")
    total = STATE.get("target_duration", 15)
    planned = plan_durations(scenes, total, video_model)
    rows = [{"scene_number": s.get("scene_number"),
             "planned": clamp_duration(video_model, d),
             "seconds": round(d, 1),
             "dialogue_words": len((s.get("dialogue") or "").split()),
             "role": s.get("role_in_story", "")}
            for s, d in zip(scenes, planned)]
    return {"rows": rows, "target": total,
            "estimated_cost": estimate_run_cost(scenes, video_model, total,
                                                STATE.get("resolution", "720p"))}


@app.post("/api/render")
def render(req: RenderRequest):
    _set_key(req.api_key)
    scenes = STATE.get("scenes") or []
    if not any(s.get("generated_image_url") for s in scenes):
        raise HTTPException(400, "Generate the storyboard first")
    campaign = STATE.get("campaign", {})
    total = STATE.get("target_duration", 15)
    cost = estimate_run_cost(scenes, req.video_model, total, req.resolution,
                             req.duration_overrides)
    if req.max_budget > 0 and cost > req.max_budget:
        raise HTTPException(402, f"Projected video cost ${cost:.2f} exceeds the "
                                 f"${req.max_budget:.2f} budget cap. Raise the cap or "
                                 f"shorten the timeline.")
    text_mode = text_mode_for(STATE.get("font_style", ""))

    def run(job):
        def progress(seg):
            job.emit("segment", scene_number=seg.get("scene_number"),
                     status=seg.get("status"), error=seg.get("error"))
        segs = render_all(scenes, campaign, req.video_model, total,
                          req.aspect_ratio, req.resolution, text_mode,
                          req.duration_overrides, on_progress=progress)
        STATE["segments"] = segs
        ok = [s["path"] for s in segs if s["status"] == "ok"]
        final = stitch_segments_natively(ok, hard_cut=bool(
            campaign.get("is_motion_graphic"))) if ok else None
        STATE["final_video"] = final
        _persist()
        return {"final": f"/media/{os.path.basename(final)}" if final else None,
                "segments": [{k: v for k, v in s.items() if k != "path"} for s in segs],
                "failed": [s["scene_number"] for s in segs if s["status"] != "ok"],
                "video_cost": cost}
    return {"job_id": jobs.start("render", run).id}


@app.post("/api/render/segment")
def render_one(req: SegmentEdit):
    _set_key(req.api_key)
    segs = STATE.get("segments") or []
    if not segs:
        raise HTTPException(400, "Nothing rendered yet")
    updated = rerender_segment(segs, req.scene_number,
                               req.prompt_override, req.image_override)
    STATE["segments"] = updated
    ok = [s["path"] for s in updated if s["status"] == "ok" and s.get("path")]
    final = stitch_segments_natively(ok, hard_cut=bool(
        (STATE.get("campaign") or {}).get("is_motion_graphic"))) if ok else None
    STATE["final_video"] = final
    _persist()
    return {"final": f"/media/{os.path.basename(final)}" if final else None,
            "segments": [{k: v for k, v in s.items() if k != "path"} for s in updated]}


# ------------------------------------------------------------------ jobs / misc
@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")

    def gen():
        import json as _json
        for ev in job.stream():
            yield f"data: {_json.dumps(ev)}\n\n"
        snapshot = {"event": "result", "status": job.status,
                    "result": job.result, "error": job.error}
        yield f"data: {_json.dumps(snapshot)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/models")
def models():
    return {"video": {k: {"label": v["label"], "price_per_sec": v["price_per_sec"],
                          "aspect_ratios": v["aspect_ratios"],
                          "resolutions": v["resolutions"]}
                      for k, v in VIDEO_MODELS.items()},
            "image": {k: {"label": v["label"], "supports_size": v["supports_size"],
                          "qualities": v.get("qualities"),
                          "resolutions": v.get("resolutions"),
                          "default_resolution": v.get("default_resolution")}
                      for k, v in IMAGE_MODELS.items()},
            "defaults": {"video": DEFAULT_VIDEO_MODEL, "image": DEFAULT_IMAGE_MODEL}}


@app.get("/api/state")
def state():
    """The LIVE working session — empty on a fresh boot. This is what every
    other endpoint actually operates on, so what this returns is what will
    be used, with no hidden carryover from a previous project."""
    return _state_payload()


@app.get("/api/session/history")
def session_history():
    """Cheap existence check for the 'restore previous session' prompt.
    Reads the on-disk autosave directly — never touches or reveals it into
    the live STATE, so a fresh boot stays genuinely fresh until the user
    explicitly opts in via /api/session/restore."""
    return {"available": session_store.has_session(),
            "age": session_store.session_age_str()}


@app.post("/api/session/restore")
def session_restore():
    """Explicit opt-in: load the last saved project into the live session."""
    STATE.clear()
    STATE.update(_load_snapshot())
    _persist()
    return _state_payload()


@app.post("/api/session/new")
def session_new():
    """Explicit opt-in the other direction: wipe the live session to start a
    clean project without restarting the server."""
    STATE.clear()
    STATE.update({"scenes": [], "segments": [], "assets": []})
    _persist()
    return _state_payload()


@app.get("/media/{name}")
def media(name: str):
    for folder in ("outputs", "downloads", UPLOAD_DIR, "outputs/frames", "outputs/chunks"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404)


app.mount("/", StaticFiles(directory="app/static", html=True), name="ui")
