"""Orchestrates multi-segment generation across models, with continuity and stitching."""
import fal_client
import requests
import os
import uuid

from app.pipeline.fusion import fuse_prompt
from app.pipeline.model_registry import MODELS
from app.pipeline.frame_extractor import extract_frames, extract_last_frame
from app.pipeline.uploader import upload_video
from app.pipeline.segment_planner import plan_segments
from app.pipeline.stitcher import stitch_segments

SEGMENTS_DIR = "segments"


def _download_result(url: str) -> str:
    """Downloads a generated segment locally so its last frame can be extracted for continuity."""
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    path = os.path.join(SEGMENTS_DIR, f"{uuid.uuid4()}.mp4")
    r = requests.get(url, timeout=120)
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _generate_single_segment(model: dict, prompt: str, duration: int, aspect_ratio: str,
                              resolution: str, video_url: str = None,
                              image_urls: list[str] = None, continuity_frame_url: str = None) -> str:
    """Calls the model endpoint once for one segment, returns the hosted output video URL."""
    args = {"prompt": prompt, "duration": str(duration) if model["input_type"] == "video" else duration}

    if model["input_type"] == "video":
        args["video_urls"] = [video_url]
        if continuity_frame_url:
            args["image_urls"] = [continuity_frame_url]
        if aspect_ratio in model["aspect_ratios"]:
            args["aspect_ratio"] = aspect_ratio
        args["resolution"] = resolution

    elif model["input_type"] == "images":
        imgs = list(image_urls or [])
        if continuity_frame_url:
            imgs = [continuity_frame_url] + imgs
        args["image_urls"] = imgs[: model["max_images"]]
        args["aspect_ratio"] = aspect_ratio if aspect_ratio in model["aspect_ratios"] else "16:9"

    result = fal_client.subscribe(model["endpoint"], arguments=args)
    output_url = result.get("video", {}).get("url")
    if not output_url:
        raise RuntimeError(f"{model['label']} returned no video output for a segment.")
    return output_url


def generate_video(model_id: str, video_url: str, local_video_path: str,
                    breakdown: str, requested_changes: str,
                    total_duration: int, aspect_ratio: str = "auto",
                    resolution: str = "720p") -> str:
    """Plans segments, generates each with continuity, stitches into final ad."""
    model = MODELS[model_id]
    segment_durations = plan_segments(total_duration, model)
    final_prompt = fuse_prompt(breakdown, requested_changes)

    all_frames = []
    if model["input_type"] == "images":
        all_frames = extract_frames(local_video_path, count=min(model["max_images"], 8))
        all_frames = [upload_video(p) for p in all_frames]

    generated_paths = []
    continuity_frame_url = None

    for i, seg_duration in enumerate(segment_durations):
        seg_prompt = final_prompt
        if i > 0:
            seg_prompt = f"Continue directly from the previous scene (@Image1 = last frame). {final_prompt}"
        elif model["input_type"] == "video":
            seg_prompt = f"Recreate @Video1's structure. {final_prompt}"

        output_url = _generate_single_segment(
            model=model,
            prompt=seg_prompt,
            duration=seg_duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            video_url=video_url if model["input_type"] == "video" else None,
            image_urls=all_frames if model["input_type"] == "images" else None,
            continuity_frame_url=continuity_frame_url,
        )

        local_seg_path = _download_result(output_url)
        generated_paths.append(local_seg_path)

        if i < len(segment_durations) - 1:
            last_frame_path = extract_last_frame(local_seg_path)
            continuity_frame_url = upload_video(last_frame_path)

    final_path = stitch_segments(generated_paths)
    return final_path