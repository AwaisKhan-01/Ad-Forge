"""Routes generation requests to the correct model backend based on model_registry."""
import fal_client
from app.pipeline.fusion import fuse_prompt
from app.pipeline.model_registry import MODELS
from app.pipeline.frame_extractor import extract_frames
from app.pipeline.uploader import upload_video


def generate_video(model_id: str, video_url: str, local_video_path: str,
                    breakdown: str, requested_changes: str,
                    duration: int, aspect_ratio: str = "auto",
                    resolution: str = "720p") -> str:
    model = MODELS[model_id]
    final_prompt = fuse_prompt(breakdown, requested_changes)

    if model["input_type"] == "video":
        prompt = f"Recreate @Video1's structure. {final_prompt}"
        args = {
            "prompt": prompt,
            "video_urls": [video_url],
            "duration": str(duration),
            "resolution": resolution,
        }
        if aspect_ratio in model["aspect_ratios"]:
            args["aspect_ratio"] = aspect_ratio

    elif model["input_type"] == "images":
        frame_paths = extract_frames(local_video_path, count=3)
        frame_urls = [upload_video(p) for p in frame_paths]  # uploader works for any file type
        prompt = final_prompt
        args = {
            "prompt": prompt,
            "image_urls": frame_urls,
            "duration": duration,
            "aspect_ratio": aspect_ratio if aspect_ratio in model["aspect_ratios"] else "16:9",
        }

    result = fal_client.subscribe(model["endpoint"], arguments=args)
    output_url = result.get("video", {}).get("url")
    if not output_url:
        raise RuntimeError(f"{model['label']} returned no video output.")

    return output_url