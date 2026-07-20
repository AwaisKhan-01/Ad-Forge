"""
Orchestrates video generation using a Temporally Isolated Hard-Cut Strategy.
Features Proportional Time Allocation, T5-Optimized Prompts, and PARALLEL Video Rendering.
Located in: app/new_pipeline/generator.py
"""
import os
import re
import uuid
import requests
import fal_client
import concurrent.futures
from typing import List, Dict

from app.new_pipeline.model_registry import MODELS
from app.new_pipeline.cost_tracker import CostTracker
from app.new_pipeline.frame_extractor import extract_last_frame 
from app.new_pipeline.uploader import upload_video
from app.new_pipeline.stitcher import stitch_segments_natively

SEGMENTS_DIR = "segments"

def _scene_speech(scene: Dict) -> str:
    """Return the spoken words for a scene.

    The analyzer emits a single 'audio' string, but frame_designer's parser
    splits it into 'spoken_dialogue' and 'sfx_music'. Prefer the split fields
    (what actually reaches this stage), falling back to the legacy 'audio' key.
    """
    spoken = (scene.get("spoken_dialogue") or "").strip()
    if spoken and spoken not in ("...", "-", "..", "n/a", "none"):
        return spoken
    # Fall back only to the legacy single "audio" field — NOT sfx_music, which is
    # background sound/ambience (e.g. "ambient", "soft piano") and must never be
    # spoken by the on-screen subject.
    legacy = (scene.get("audio") or "").strip()
    if legacy and legacy not in ("...", "-", "..", "n/a", "none"):
        return legacy
    return ""


def get_semantic_duration(ts_str: str) -> float:
    try:
        parts = ts_str.replace(" ", "").split('-')
        def time_to_sec(t_str):
            if ':' in t_str:
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        start_sec = time_to_sec(parts[0])
        if len(parts) > 1:
            return max(1.0, time_to_sec(parts[1]) - start_sec)
        return 3.0 
    except Exception:
        return 3.0

def plan_dynamic_segments(storyboard_frames: List[Dict], total_target_duration: int) -> List[float]:
    if not storyboard_frames:
        return []
    
    base_weights = []
    
    # --- SURGICAL FIX: Speech-Aware Proportional Weighting ---
    for s in storyboard_frames:
        sem_dur = get_semantic_duration(s.get("timestamp", ""))

        # Calculate how long the dialogue physically takes to speak (0.4s per word).
        # The parser splits audio into spoken_dialogue / sfx_music, so read those —
        # the legacy "audio" key was always empty and made this branch dead code.
        scene_audio = _scene_speech(s)
        word_count = len(scene_audio.split()) if scene_audio and "music" not in scene_audio.lower() else 0
        audio_time = word_count * 0.4
        
        # The weight of this scene is whichever is larger: the VLM's timestamp, or the physical speech requirement
        base_weights.append(max(sem_dur, audio_time))
        
    total_weight = sum(base_weights)
    
    allocated_durations = []
    if total_weight <= 0:
        base = float(total_target_duration) / len(storyboard_frames)
        return [base] * len(storyboard_frames)
        
    for w in base_weights:
        # We STRICTLY map the weights back to the exact target duration. 
        # This prevents video bloat while guaranteeing dialogue-heavy scenes get the seconds they need.
        allocated = (w / total_weight) * float(total_target_duration)
        allocated_durations.append(allocated)
        
    return allocated_durations

def sanitize_video_prompt(raw_desc: str) -> str:
    clean = raw_desc.replace('"', '').replace("'", "")
    
    # --- SURGICAL FIX: Safety Sanitizer for False Positives ---
    # Swaps words that trigger Gemini's Violence/NSFW filters with safe alternatives
    safety_replacements = {
        "ripped": "distressed",
        "naked": "bare",
        "shoot": "film",
        "shot": "filmed",
        "kill": "end",
        "blood": "red liquid",
        "bomb": "burst",
        "explosive": "dynamic",
        "gun": "prop",
        "weapon": "prop"
    }
    
    for bad_word, safe_word in safety_replacements.items():
        # Uses regex to replace the exact word bounds, case-insensitive
        clean = re.sub(rf'(?i)\b{bad_word}\b', safe_word, clean)
        
    # Remove only true editing/meta directives that confuse a text-to-video model.
    # NOTE: do NOT strip content words like "transform"/"scene"/"visual" — they are
    # often part of the actual description (e.g. "transform the cream tube") and
    # deleting them mangles the prompt. Match whole meta phrases only.
    meta_phrases = [
        r"cut to", r"\bcut\b", r"\bfade (?:in|out)\b", r"\bfade\b",
        r"\btransition(?:s|ing)?\b", r"\bvoiceover\b",
    ]
    for phrase in meta_phrases:
        clean = re.sub(rf'(?i){phrase}', '', clean)

    # Collapse any double spaces left behind by the removals.
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    if clean and not clean.endswith('.'):
        clean += '.'
    return clean

def _download_result(url: str) -> str:
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    path = os.path.join(SEGMENTS_DIR, f"{uuid.uuid4()}.mp4")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f: 
            f.write(r.content)
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to download video chunk: {e}")

def _process_single_video_segment(idx: int, scene: Dict, seg_duration: float, total_scenes: int, 
                                  final_instructions: str, model: dict, aspect_ratio: str, 
                                  resolution: str, production_type: str) -> Dict:
    # Prefer the cleaned description frame_designer produced (meta/sequence text
    # stripped); fall back to the raw VLM text only if the image stage was skipped.
    raw_v_desc = scene.get("visual_prompt_sterilized") or scene.get("visual_prompt", "")

    # --- Bind role references cleanly using the required <IMAGE_REF_0> tag format ---
    # Handle both the original #assetN tags and the #imageN tags the image stage
    # rewrote them into, so neither leaks into the video prompt as literal text.
    raw_v_desc = re.sub(r"(?i)\s*\(?#(?:asset|image)\s*\d+\)?", " <IMAGE_REF_0> ", raw_v_desc)
    
    v_desc = sanitize_video_prompt(raw_v_desc)
    m_desc = sanitize_video_prompt(scene.get("motion", ""))
    img_url = scene.get("generated_image_url")
    scene_audio = _scene_speech(scene)
    
    isolated_prompt = f"{v_desc} Camera motion: {m_desc} Cinematic lighting, clean professional render."
    
    # --- SURGICAL FIX: Enforce strict proportional duration pacing to eliminate timeline bloat ---
    true_duration = seg_duration
    
    if scene_audio and "On-Screen Actor" in production_type:
        if "music" in scene_audio.lower() or "score" in scene_audio.lower() or scene_audio in ["", "..."]:
            isolated_prompt += " Subject is silent, natural ambient movement. No text on screen."
        else:
            safe_audio = scene_audio.replace('"', '').replace("'", "")
            isolated_prompt += (
                f' The subject speaks these exact words in a natural, warm, conversational human voice: "{safe_audio}". '
                f'VOICE DIRECTION: Authentic real-person delivery with natural intonation, breath, and pacing — '
                f'relaxed and expressive, NOT robotic, monotone, flat, or synthetic. Match the tone and vocal '
                f'identity of the reference person; keep the same speaker throughout. Accurate lip-sync. '
                f'CRITICAL TIMING: Start the clip cleanly without text; typography animates on screen only as the subject begins speaking.'
            )
        
    elif scene_audio and "Motion Graphic" in production_type:
        isolated_prompt += " Smooth kinetic motion. Text animates dynamically."
            
    if idx == total_scenes - 1 and final_instructions:
        isolated_prompt += f" {sanitize_video_prompt(final_instructions)}"

    # Protect against schema validation failures while honoring the exact contract limits (Range 3-10)
    safe_integer_duration = max(3, min(10, int(round(true_duration))))

    args = {
        "prompt": isolated_prompt.strip(), 
        "duration": safe_integer_duration, 
        "aspect_ratio": aspect_ratio
    }
    
    if img_url:
        args["image_url"] = img_url
        args["image_urls"] = [img_url]

    if model.get("input_type") == "video": 
        args["resolution"] = resolution
    
    print(f"\n      [DISPATCHING] Scene {idx+1}/{total_scenes} -> {safe_integer_duration}s requested")
    print(f"      [PROMPT SENT]: {args['prompt']}")

    # NOTE: this function NEVER raises. A single chunk failing must not tear down
    # the whole run and discard the chunks the user already paid for. Instead we
    # return a result dict with status="failed" that carries the real error and
    # the exact args sent, so the UI can show WHY it failed and the user can fix
    # the cause and re-send just this one segment (see regenerate_single_segment).
    base_result = {
        "idx": idx,
        "duration": safe_integer_duration,
        "endpoint": model["endpoint"],
        "args": args,           # exact payload sent — lets the user inspect/edit and retry
        "scene_number": scene.get("scene_number", idx + 1),
    }
    try:
        result = fal_client.subscribe(model["endpoint"], arguments=args)
        out_url = result.get("video", {}).get("url")
        if not out_url:
            raise RuntimeError("Model returned no video URL (empty result). "
                               "Often a silent content-policy rejection of the prompt.")
        local_path = _download_result(out_url)
        print(f"      [COMPLETED] Scene {idx+1}")
        return {**base_result, "status": "ok", "path": local_path, "error": None}
    except Exception as e:
        error_msg = str(e) or e.__class__.__name__
        print(f"      [FAILED] Scene {idx+1}: {error_msg}")
        return {**base_result, "status": "failed", "path": None, "error": error_msg}

def generate_final_video(storyboard_frames: List[Dict], final_instructions: str, model_id: str, 
                         cost_tracker: CostTracker, total_duration: int, aspect_ratio: str, 
                         resolution: str, raw_script_text: str, production_type: str = "Narrative") -> str:
    
    model = MODELS[model_id]
    total_scenes = len(storyboard_frames)
    
    if not storyboard_frames: 
        raise RuntimeError("No valid scenes found in the structured storyboard.")
        
    scene_durations = plan_dynamic_segments(storyboard_frames, total_duration)

    print("\n[STORYBOARD ENGINE] Initializing Parallel Video Rendering Matrix...")
    segment_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, scene in enumerate(storyboard_frames):
            futures.append(executor.submit(
                _process_single_video_segment,
                idx, scene, scene_durations[idx], total_scenes,
                final_instructions, model, aspect_ratio, resolution, production_type
            ))

        for future in concurrent.futures.as_completed(futures):
            # _process_single_video_segment never raises; it returns a status dict.
            segment_results.append(future.result())

    segment_results.sort(key=lambda x: x["idx"])

    # Attach the model_id so a later manual re-render knows which engine to use,
    # and bill only for the chunks that actually succeeded.
    for res in segment_results:
        res["model_id"] = model_id
        if res["status"] == "ok":
            cost_tracker.add_segment(model_id, res["duration"], resolution)

    return _assemble_from_segments(segment_results)


def _assemble_from_segments(segment_results: List[Dict]) -> Dict:
    """Stitch whatever chunks succeeded and report on any that failed.

    Returns a dict (never just a path) so the UI can render a partial video AND
    show exactly which scenes failed and why. The full per-segment state is
    returned so the user can fix a bad chunk and re-render only that one.
    """
    segment_results = sorted(segment_results, key=lambda x: x["idx"])
    ok = [r for r in segment_results if r["status"] == "ok" and r.get("path")]
    failed = [r for r in segment_results if r["status"] != "ok"]

    final_video_path = None
    if ok:
        print("\n[STITCHING] Compiling successful scenes into master timeline...")
        final_video_path = stitch_segments_natively([r["path"] for r in ok])

    if failed:
        lines = [f"⚠ {len(failed)} of {len(segment_results)} scene(s) failed. "
                 f"Rendered {len(ok)} successfully."]
        for r in failed:
            lines.append(f"  • Scene {r.get('scene_number', r['idx'] + 1)}: {r['error']}")
        lines.append("\nFix the cause (edit the prompt / swap the image), then "
                     "re-render only the failed scene — successful scenes are kept.")
        status_report = "\n".join(lines)
    else:
        status_report = f"✅ All {len(ok)} scenes rendered successfully."

    return {
        "final_video_path": final_video_path,
        "segments": segment_results,   # persisted so a single scene can be re-rendered
        "status_report": status_report,
        "failed_scene_numbers": [r.get("scene_number", r["idx"] + 1) for r in failed],
    }


def regenerate_single_segment(segments: List[Dict], scene_number, cost_tracker: CostTracker,
                              resolution: str, prompt_override: str = "",
                              image_url_override: str = "") -> Dict:
    """Re-render exactly ONE previously-failed (or any) segment, then re-stitch.

    This is NOT an automatic retry: the caller supplies a fix (an edited prompt
    and/or a different image) so the segment isn't re-sent with the same args
    that failed. All other segments are left untouched and are not re-billed.
    """
    segments = [dict(s) for s in segments]  # shallow copy so we don't mutate caller state
    target = next((s for s in segments if str(s.get("scene_number")) == str(scene_number)
                   or str(s["idx"] + 1) == str(scene_number)), None)
    if target is None:
        return _assemble_from_segments(segments) | {
            "status_report": f"Scene {scene_number} not found in the rendered set."}

    args = dict(target["args"])
    if prompt_override and prompt_override.strip():
        args["prompt"] = prompt_override.strip()
    if image_url_override and image_url_override.strip():
        args["image_url"] = image_url_override.strip()
        args["image_urls"] = [image_url_override.strip()]

    print(f"\n[RE-RENDER] Scene {scene_number} with user-corrected args...")
    print(f"      [PROMPT SENT]: {args.get('prompt')}")
    try:
        result = fal_client.subscribe(target["endpoint"], arguments=args)
        out_url = result.get("video", {}).get("url")
        if not out_url:
            raise RuntimeError("Model returned no video URL (empty result). "
                               "Often a silent content-policy rejection of the prompt.")
        local_path = _download_result(out_url)
        target.update(status="ok", path=local_path, error=None, args=args)
        cost_tracker.add_segment(target.get("model_id", ""), target["duration"], resolution)
        print(f"      [COMPLETED] Scene {scene_number} re-render")
    except Exception as e:
        target.update(status="failed", path=None, error=str(e) or e.__class__.__name__, args=args)
        print(f"      [FAILED] Scene {scene_number} re-render: {target['error']}")

    return _assemble_from_segments(segments)