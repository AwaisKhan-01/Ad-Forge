"""Frame designer — turns each blueprint scene into a storyboard frame that is
TAILORED TO THE USER'S PRODUCT.

v1's core flaw: it extracted the reference keyframe and lightly edited it, so
the storyboard stayed about the REFERENCE's product. v2 inverts the roles:
  #image1        = reference keyframe  -> STYLE ONLY (composition/light/grade)
  #image2..N     = user's assets       -> the actual CONTENT
and the prompt (built centrally in prompt_engine) instructs replace-not-retouch,
driven by the analyzer's per-scene adapt.keep / adapt.replace brief.
"""
import os
import subprocess
import uuid
import concurrent.futures
from typing import Dict, List, Optional

import fal_client

from app.new_pipeline.model_registry import IMAGE_MODELS, DEFAULT_IMAGE_MODEL
from app.new_pipeline.prompt_engine import build_frame_prompt, validate_frame_prompt
from app.new_pipeline.uploader import upload_video

FRAME_DIR = "outputs/frames"

TYPOGRAPHY_STYLES = {
    "KEEP_EXISTING_TEXT": None,     # third state: preserve the reference's own type
    "NO_TEXT": None,                # ban all text, including original
    # statement styles
    "chunky_display": "bold chunky hand-drawn display type, thick solid weights",
    "elegant_script_mix": "high-contrast elegant script mixed with clean geometric sans",
    "streetwear_graffiti": "distressed stencil streetwear type, crisp edges",
    "brutalist_geometric": "modular blocky brutalist type, uniform stroke",
    "retro_3d_editorial": "vintage 70s dimensional editorial lettering",
    # simpler, quieter styles
    "clean_sans": "clean minimal modern sans-serif, medium weight, generous letter-spacing, understated",
    "minimal_serif": "refined editorial serif, elegant and quiet, lots of whitespace",
    "handwritten_casual": "casual handwritten marker lettering, friendly and human",
}


def text_mode_for(font_style: str) -> str:
    if font_style == "KEEP_EXISTING_TEXT":
        return "keep"
    if font_style == "NO_TEXT":
        return "none"
    return "add"


def parse_keyframe_seconds(kf: str) -> float:
    try:
        kf = (kf or "0:00").strip()
        if ":" in kf:
            m, s = kf.split(":")
            return int(m) * 60 + float(s)
        return float(kf)
    except Exception:
        return 0.0


def extract_frame(video_path: str, ts: float) -> Optional[str]:
    """Frame-accurate grab: -ss AFTER -i (output seek) or we land on a prior
    I-frame and the keyframe shows the wrong moment in fast-paced ads."""
    os.makedirs(FRAME_DIR, exist_ok=True)
    out = os.path.join(FRAME_DIR, f"{uuid.uuid4().hex[:10]}.png")
    cmd = ["ffmpeg", "-i", video_path, "-ss", f"{ts:.3f}",
           "-frames:v", "1", "-q:v", "2", "-y", out]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return out if os.path.exists(out) else None
    except Exception as e:
        print(f"[FRAMES] extract failed @{ts:.2f}s: {e}")
        return None


def design_frame(scene: Dict, campaign: Dict, asset_urls: List[str],
                 reference_path: str, aspect_ratio: str, font_style: str,
                 image_model: str = DEFAULT_IMAGE_MODEL, image_size: str = "",
                 image_quality: str = "high", resolution: str = "",
                 prompt_override: str = "", images_override: Optional[List[str]] = None) -> Dict:
    """Generate ONE storyboard frame. Never raises — failures are recorded on
    the scene so one bad frame can't kill the batch."""
    text_mode = text_mode_for(font_style)
    mg = bool(campaign.get("is_motion_graphic"))

    # 1. Reference keyframe = style anchor (#image1).
    ts = parse_keyframe_seconds(scene.get("keyframe_time", ""))
    frame_path = extract_frame(reference_path, ts)
    if not frame_path:
        scene.update(generated_image_url=None, image_error="keyframe extraction failed")
        return scene
    style_url = upload_video(frame_path)

    # 2. User assets = content (#image2..N), selected 1-based by the blueprint.
    # bound_nums tracks, IN ORDER, which asset number landed at each position —
    # e.g. bound_nums=[3,1] means Asset 3 is #image2 and Asset 1 is #image3.
    # Without this the prompt's "Asset N" wording can't be tied to a real tag.
    picked, bound_nums = [], []
    for n in scene.get("assigned_assets", []) or []:
        if 1 <= n <= len(asset_urls):
            picked.append(asset_urls[n - 1])
            bound_nums.append(n)
    if not picked:
        picked = asset_urls[:2]         # sensible default: brand's main assets
        bound_nums = list(range(1, len(picked) + 1))
    image_urls = images_override or ([style_url] + picked)
    # images_override bypasses our own asset bookkeeping (a manual re-render),
    # so there's no reliable Asset-N -> #imageK mapping to apply.
    asset_binding = bound_nums if not images_override else None

    # 3. Prompt from the central engine (or the user's exact edit).
    prompt = prompt_override.strip() if prompt_override and prompt_override.strip() else \
        build_frame_prompt(scene, campaign, has_user_assets=bool(picked),
                           text_mode=text_mode,
                           font_style_desc=TYPOGRAPHY_STYLES.get(font_style) or "",
                           asset_binding=asset_binding)

    # 3b. Pre-flight check — catches a dangling #imageN or an unbound "Asset N"
    # BEFORE spending on a generation call, instead of silently producing a
    # frame that ignores the reference it couldn't resolve.
    prompt, warnings = validate_frame_prompt(prompt, len(image_urls))
    scene["prompt_warning"] = "; ".join(warnings) if warnings else None
    if warnings:
        print(f"[FRAMES] scene {scene.get('scene_number')} prompt warning: {scene['prompt_warning']}")

    # 4. Strength: replacing content needs room; preserving lettering needs restraint.
    strength = 0.85 if mg else (0.6 if picked else 0.3)
    if text_mode == "keep":
        strength = min(strength, 0.35)

    cfg = IMAGE_MODELS.get(image_model, IMAGE_MODELS[DEFAULT_IMAGE_MODEL])
    args = {"prompt": prompt, "image_urls": image_urls}
    if cfg["supports_size"]:
        args["image_size"] = image_size or cfg["sizes"][0]
        args["quality"] = image_quality or "auto"
    else:
        args["aspect_ratio"] = aspect_ratio
        args["strength"] = strength
        if cfg.get("resolutions"):
            args["resolution"] = resolution if resolution in cfg["resolutions"] \
                else cfg.get("default_resolution", "1K")

    scene["image_args"], scene["image_endpoint"] = args, cfg["endpoint"]
    try:
        res = fal_client.subscribe(cfg["endpoint"], arguments=args)
        url = (res.get("images") or [{}])[0].get("url")
        scene["generated_image_url"] = url
        scene["image_error"] = None if url else "model returned no image"
    except Exception as e:
        scene["generated_image_url"] = None
        scene["image_error"] = str(e)
        print(f"[FRAMES] scene {scene.get('scene_number')} failed: {e}")
    return scene


def design_storyboard(scenes: List[Dict], campaign: Dict, asset_paths: List[str],
                      reference_path: str, aspect_ratio: str, font_style: str,
                      image_model: str = DEFAULT_IMAGE_MODEL, image_size: str = "",
                      image_quality: str = "high", resolution: str = "",
                      on_progress=None) -> List[Dict]:
    """All frames in parallel; order restored by scene_number."""
    asset_urls = [a if str(a).startswith("http") else upload_video(a)
                  for a in (asset_paths or []) if a]
    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(design_frame, s, campaign, asset_urls, reference_path,
                            aspect_ratio, font_style, image_model, image_size,
                            image_quality, resolution): s for s in scenes}
        for f in concurrent.futures.as_completed(futs):
            result = f.result()
            done.append(result)
            if on_progress:
                on_progress(result)
    done.sort(key=lambda s: int(s.get("scene_number", 0)))
    return done
