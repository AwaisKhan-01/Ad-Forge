"""Orchestrates multi-segment generation across models, with continuity, vision-grounded prompts, and stitching."""
import fal_client
import requests
import os
import uuid

from app.pipeline.fusion import fuse_prompt, fuse_prompt_images
from app.pipeline.frame_captioner import caption_frames
from app.pipeline.model_registry import MODELS
from app.pipeline.frame_extractor import extract_frames_in_range, extract_last_frame, _get_duration
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
    args = {"prompt": prompt}

    if model["input_type"] == "video":
        args["duration"] = str(duration)
        args["video_urls"] = [video_url]
        combined_images = list(image_urls or [])
        if continuity_frame_url:
            combined_images = [continuity_frame_url] + combined_images
        if combined_images:
            args["image_urls"] = combined_images[: model["max_images"]]
        if aspect_ratio in model["aspect_ratios"]:
            args["aspect_ratio"] = aspect_ratio
        args["resolution"] = resolution

    elif model["input_type"] == "images":
        args["duration"] = duration
        combined_images = list(image_urls or [])
        if continuity_frame_url:
            combined_images = [continuity_frame_url] + combined_images
        args["image_urls"] = combined_images[: model["max_images"]]
        args["aspect_ratio"] = aspect_ratio if aspect_ratio in model["aspect_ratios"] else "16:9"

    result = fal_client.subscribe(model["endpoint"], arguments=args)
    output_url = result.get("video", {}).get("url")
    if not output_url:
        raise RuntimeError(f"{model['label']} returned no video output for a segment.")
    return output_url


def generate_video(model_id: str, video_url: str, local_video_path: str,
                    breakdown: str, requested_changes: str,
                    total_duration: int, aspect_ratio: str = "auto",
                    resolution: str = "720p", user_image_urls: list[str] = None,
                    user_image_note: str = "") -> str:
    """Plans segments, generates each with continuity + vision-grounded prompts, stitches into final ad."""
    model = MODELS[model_id]
    segment_durations = plan_segments(total_duration, model)

    combined_changes = requested_changes
    if user_image_note:
        combined_changes = f"{requested_changes}. {user_image_note}".strip(". ")

    user_images = list(user_image_urls or [])
    source_duration = _get_duration(local_video_path) if model["input_type"] == "images" else 0

    generated_paths = []
    continuity_frame_url = None
    time_cursor = 0.0  # tracks position within the SOURCE reference video per segment

    for i, seg_duration in enumerate(segment_durations):
        segment_reference_images = list(user_images)  # user images apply to every segment

        if model["input_type"] == "images":
            # Map this segment's slice proportionally onto the source video's timeline
            proportion = seg_duration / total_duration
            slice_start = time_cursor
            slice_end = min(time_cursor + proportion * source_duration, source_duration)
            time_cursor = slice_end

            remaining_slots = max(model["max_images"] - len(user_images), 1)
            extracted_paths = extract_frames_in_range(
                local_video_path, slice_start, slice_end, count=min(remaining_slots, 8)
            )
            segment_reference_images += [upload_video(p) for p in extracted_paths]

            # Ground the prompt in what these frames ACTUALLY show, not invented content
            captions = caption_frames(segment_reference_images)
            final_prompt = fuse_prompt_images(captions, combined_changes)

        else:
            final_prompt = fuse_prompt(breakdown, combined_changes, model_input_type="video")

        if i == 0:
            seg_prompt = f"Recreate @Video1's structure. {final_prompt}" if model["input_type"] == "video" else final_prompt
        else:
            seg_prompt = f"Continue directly from the previous scene. {final_prompt}"

        output_url = _generate_single_segment(
            model=model,
            prompt=seg_prompt,
            duration=seg_duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            video_url=video_url if model["input_type"] == "video" else None,
            image_urls=segment_reference_images,
            continuity_frame_url=continuity_frame_url,
        )

        local_seg_path = _download_result(output_url)
        generated_paths.append(local_seg_path)

        if i < len(segment_durations) - 1:
            last_frame_path = extract_last_frame(local_seg_path)
            continuity_frame_url = upload_video(last_frame_path)

    final_path = stitch_segments(generated_paths)
    return final_path