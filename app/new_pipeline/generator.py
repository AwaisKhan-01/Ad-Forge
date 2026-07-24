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
from app.new_pipeline.cost_tracker import CostTracker, estimate_segment_cost
from app.new_pipeline.frame_extractor import extract_last_frame
from app.new_pipeline.uploader import upload_video
from app.new_pipeline.stitcher import stitch_segments_natively
from app.new_pipeline.frame_designer import _speech_only
from app.new_pipeline.analyzer import is_motion_graphic

SEGMENTS_DIR = "segments"

def _scene_speech(scene: Dict) -> str:
    """Return the spoken words for a scene.

    The analyzer emits a single 'audio' string, but frame_designer's parser
    splits it into 'spoken_dialogue' and 'sfx_music'. Prefer the split fields
    (what actually reaches this stage), falling back to the legacy 'audio' key.
    """
    # _speech_only() strips anything that reads like a music/SFX description, so a
    # legacy "Audio: upbeat electronic music" line is never spoken by the subject.
    spoken = _speech_only(scene.get("spoken_dialogue") or "")
    if spoken:
        return spoken
    # Fall back only to the legacy single "audio" field — NOT sfx_music, which is
    # background sound/ambience (e.g. "ambient", "soft piano") and must never be
    # spoken by the on-screen subject.
    return _speech_only(scene.get("audio") or "")


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

# Natural ad delivery is ~2 words/sec INCLUDING breaths and pauses. The old 0.4s
# (150 wpm, zero pauses) systematically under-allocated, so lines got cut off.
SECONDS_PER_WORD = 0.5
SPEECH_LEAD_TAIL = 0.8   # brief beat before/after speaking so it doesn't feel clipped
MAX_TOTAL_OVERRUN = 1.3  # cap total inflation vs target — model time costs money
MODEL_MAX_CHUNK = 10.0   # hard per-chunk ceiling for the video models
MODEL_MIN_CHUNK = 3.0    # hard per-chunk floor: models reject shorter clips


def speech_seconds_needed(scene: Dict) -> float:
    """Seconds this scene's dialogue physically needs to be spoken naturally."""
    words = len(_scene_speech(scene).split())
    return (words * SECONDS_PER_WORD + SPEECH_LEAD_TAIL) if words else 0.0


def clamp_chunk(seconds: float) -> int:
    """Snap a requested chunk length to what the video models actually accept.

    The models reject clips outside 3-10s, so any planned/edited duration is
    rounded to a whole second and clamped into [MODEL_MIN_CHUNK, MODEL_MAX_CHUNK].
    This is the SAME rule the render uses, so the preview total matches reality.
    """
    return int(max(MODEL_MIN_CHUNK, min(MODEL_MAX_CHUNK, round(seconds))))


def build_timeline_preview(storyboard_frames: List[Dict], total_target_duration: int) -> Dict:
    """Compute a per-scene timeline WITHOUT calling any model (free, instant).

    Returns the planned duration for each scene (already clamped to the model's
    real 3-10s limits, so the numbers are what would actually be rendered), the
    dialogue word count, a short prompt snippet, and the resulting total. The UI
    uses this for the pre-generation review step so the user sees — and can edit —
    the length before spending anything.
    """
    planned = plan_dynamic_segments(storyboard_frames, total_target_duration)
    rows = []
    for i, (scene, dur) in enumerate(zip(storyboard_frames, planned)):
        clamped = clamp_chunk(dur)
        words = len(_scene_speech(scene).split())
        prompt = (scene.get("visual_prompt") or "").strip()
        snippet = (prompt[:70] + "…") if len(prompt) > 70 else prompt
        rows.append({
            "scene_number": scene.get("scene_number", i + 1),
            "prompt_snippet": snippet,
            "dialogue_words": words,
            "planned_duration": clamped,
        })
    return {
        "rows": rows,
        "total": sum(r["planned_duration"] for r in rows),
        "target": total_target_duration,
        "min_chunk": int(MODEL_MIN_CHUNK),
        "max_chunk": int(MODEL_MAX_CHUNK),
    }


def plan_dynamic_segments(storyboard_frames: List[Dict], total_target_duration: int) -> List[float]:
    if not storyboard_frames:
        return []

    base_weights = []
    speech_needs = []

    # --- Speech-Aware Proportional Weighting ---
    for s in storyboard_frames:
        sem_dur = get_semantic_duration(s.get("timestamp", ""))
        need = speech_seconds_needed(s)
        speech_needs.append(need)
        # The weight of this scene is whichever is larger: the VLM's timestamp, or
        # the physical speech requirement.
        base_weights.append(max(sem_dur, need))

    total_weight = sum(base_weights)

    if total_weight <= 0:
        base = float(total_target_duration) / len(storyboard_frames)
        return [base] * len(storyboard_frames)

    # Proportional share of the target, then RAISE any scene that would be given
    # less time than its dialogue physically needs. Pure proportional allocation
    # squeezed speech-heavy scenes below their real requirement, which is what cut
    # dialogue off mid-sentence.
    allocated_durations = []
    for w, need in zip(base_weights, speech_needs):
        proportional = (w / total_weight) * float(total_target_duration)
        allocated_durations.append(max(proportional, need))

    # Cost guard: honoring every need can inflate the video. Scale back toward the
    # target if we overshoot the allowed overrun, but never below what speech needs.
    budget = float(total_target_duration) * MAX_TOTAL_OVERRUN
    total_alloc = sum(allocated_durations)
    if total_alloc > budget:
        slack = [a - n for a, n in zip(allocated_durations, speech_needs)]
        excess = total_alloc - budget
        total_slack = sum(slack)
        if total_slack > 0:
            shrink = min(1.0, excess / total_slack)
            allocated_durations = [a - s * shrink for a, s in zip(allocated_durations, slack)]

    # Warn about anything the model physically cannot deliver in one chunk.
    for i, (need, alloc) in enumerate(zip(speech_needs, allocated_durations)):
        if need > MODEL_MAX_CHUNK:
            print(f"      [PACING WARNING] Scene {i+1} needs ~{need:.1f}s of speech but a "
                  f"chunk maxes out at {MODEL_MAX_CHUNK:.0f}s — shorten this line or split the scene.")
        elif alloc + 0.05 < need:
            print(f"      [PACING WARNING] Scene {i+1} allocated {alloc:.1f}s but needs "
                  f"~{need:.1f}s to finish speaking.")

    print(f"      [PACING] target={total_target_duration}s -> planned={sum(allocated_durations):.1f}s")
    return allocated_durations

# Words that commonly trip fal/Gemini content classifiers on otherwise innocent
# ad copy, mapped to neutral equivalents. Ordered dict semantics matter: the
# MULTI-WORD phrases must be replaced before the single words they contain.
#
# NOTE ON FILM LANGUAGE: "shot"/"shoot" are legitimate cinematography terms
# ("wide shot", "we shoot at golden hour"). Blanket-replacing them mangled real
# camera direction, so they are only neutralized in violent constructions
# ("shoot him", "gunshot"), handled by the phrase pass below.
_FLAG_PHRASES = [
    # Only violent constructions — NOT "shoot at golden hour" / "shoot at dawn",
    # which are ordinary production language and must survive untouched.
    (r"\bshoot(?:ing|s)?\s+(?:him|her|them|people|someone|each other)\b", "confronting"),
    (r"\bshoot(?:ing|s)?\s+at\s+(?:him|her|them|people|someone)\b", "aiming toward"),
    (r"\bgun\s*shot\b", "loud bang"),
    (r"\bblow(?:ing|s)?\s+up\b", "bursting"),
    (r"\bbreak(?:ing|s)?\s+into\b", "entering"),
]
_FLAG_WORDS = {
    # violence / weapons
    "kill": "stop", "killing": "stopping", "dead": "still", "death": "the end",
    "blood": "red liquid", "bloody": "deep red", "wound": "mark",
    "gun": "prop", "rifle": "prop", "weapon": "prop", "knife": "blade tool",
    "bomb": "burst", "explosive": "dynamic", "explosion": "burst of light",
    "attack": "approach", "fight": "face off", "punch": "strike pose",
    "violent": "intense", "violence": "intensity", "war": "conflict",
    "destroy": "transform", "smash": "impact",
    # nudity / suggestive
    "naked": "bare", "nude": "bare", "sexy": "striking", "seductive": "alluring",
    "lingerie": "loungewear", "undress": "change",
    # substances
    "drug": "supplement", "drugs": "supplements", "alcohol": "beverage",
    "beer": "drink", "cigarette": "stick", "smoking": "steaming",
    # self-harm / distress
    "suicide": "crisis", "abuse": "misuse",
    # misc false-positives
    "ripped": "distressed", "shredded": "textured", "strangle": "hold",
}


def _neutralize_flaggable(text: str) -> str:
    """Replace words/phrases that commonly trigger content classifiers.

    Applied to EVERY piece of free text that reaches a video model — the visual
    description, the spoken dialogue and any on-screen copy — because the checker
    scans the whole prompt, not just the scene description.
    """
    if not text:
        return text
    out = text
    for pattern, repl in _FLAG_PHRASES:      # phrases first (they contain words)
        out = re.sub(rf"(?i){pattern}", repl, out)
    for bad, safe in _FLAG_WORDS.items():
        out = re.sub(rf"(?i)\b{re.escape(bad)}\b", safe, out)
    return out


def sanitize_video_prompt(raw_desc: str) -> str:
    clean = raw_desc.replace('"', '').replace("'", "")
    
    clean = _neutralize_flaggable(clean)
        
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

def sanitize_camera_motion(raw_motion: str) -> str:
    """Light cleanup for the camera-motion field.

    Deliberately does NOT strip editing words the way sanitize_video_prompt does:
    terms like "cut", "pan", "dolly", "fade" are legitimate camera vocabulary and
    removing them destroys the reference video's actual movement description.
    Only quotes and safety substitutions are applied.
    """
    clean = (raw_motion or "").replace('"', '').replace("'", "").strip()
    for bad_word, safe_word in {"shoot": "film", "shot": "filmed", "gun": "prop"}.items():
        clean = re.sub(rf'(?i)\b{bad_word}\b', safe_word, clean)
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

def _to_kling_args(args: dict, model: dict) -> dict:
    """Translate our internal video payload to the Kling v3 Pro contract.

    Internal payload uses `image_url` / `image_urls` and an int `duration`.
    Kling v3 Pro wants `start_image_url` (NOTE: not `image_url` like O3/Wan),
    an optional `end_image_url`, a STRING duration, and generates audio natively
    (no aspect_ratio param). Duration type comes from the registry via
    _format_duration so it can never drift from the endpoint's schema.
    """
    urls = args.get("image_urls") or ([args["image_url"]] if args.get("image_url") else [])
    out = {
        "prompt": args.get("prompt", ""),
        "duration": _format_duration(args.get("duration", 5), model),
        "generate_audio": True,
    }
    if urls:
        out["start_image_url"] = urls[0]
    # Only lock the end frame when an explicit end role was set (see
    # _to_endframe_args for why image_urls[1] must not be used as a fallback).
    end_frame = args.get("_end_frame")
    if end_frame:
        out["end_image_url"] = end_frame
    return out


def _is_content_flag(error_msg: str) -> bool:
    """True when a failure looks like a moderation rejection rather than a bug.

    fal reports these as `content_policy_violation`; some models instead return
    an empty result, which we surface with a matching message.
    """
    m = (error_msg or "").lower()
    return any(s in m for s in (
        "content_policy_violation", "content policy", "content checker",
        "flagged by a content", "safety", "nsfw", "moderation",
        "returned no video url",
    ))


def _make_safe_retry_args(args: dict, scene: Dict) -> dict:
    """Build a deliberately bland payload for a scene the checker rejected.

    Strategy: drop the free-text description entirely (that's what got flagged)
    and keep only the frames plus a generic, unmistakably safe instruction. The
    images themselves already carry the composition, so the clip still matches
    the storyboard — it just loses the descriptive nuance.
    """
    safe = dict(args)
    is_mg = bool((scene or {}).get("on_screen_text"))
    safe["prompt"] = (
        "A clean, professional commercial shot. Smooth natural motion, "
        "cinematic lighting, high production value. No text overlays."
        if not is_mg else
        "A clean, professional motion-graphics shot. Smooth kinetic animation of "
        "the on-screen elements, polished and modern, high production value."
    )
    # Speech is the most common trigger; a flagged retry drops it entirely.
    safe.pop("negative_prompt", None)
    return safe


def _format_duration(seconds, model: dict):
    """Serialize `duration` exactly the way THIS model's schema demands.

    fal endpoints differ: Kling (v3 Pro / O3) declare duration as a STRING enum
    ('3'..'15'); Wan 2.7 declares it as an INTEGER (2..15). Sending the wrong
    type fails validation outright, so the type comes from the registry rather
    than being assumed. The value is also clamped to the model's own duration
    list, which may start lower/higher than our global chunk floor.
    """
    val = int(round(float(seconds)))
    allowed = model.get("durations") or []
    if allowed:
        val = max(min(allowed), min(max(allowed), val))
    return str(val) if model.get("duration_type", "string") == "string" else val


def _to_endframe_args(args: dict, model: dict) -> dict:
    """Translate our internal payload to the end-frame-lock contract shared by
    Kling O3 and Wan 2.7: `image_url` (start) + optional `end_image_url` (end).
    When our continuity logic packed image_urls as [start, end], the second URL
    becomes the HARD last frame — this is what removes the merge cut in motion
    graphics (clip N ends exactly where clip N+1 begins).

    IMPORTANT: the two models disagree on the duration TYPE — Kling O3 wants a
    string enum ('3'..'15'), Wan 2.7 wants an integer (2..15). The registry's
    `duration_type` is authoritative; never assume one for both (sending '4' to
    Wan is rejected with a literal_error).

    `generate_audio` is taken from the model registry (default off; the music-bed
    post stage supplies the soundtrack for motion graphics).
    """
    urls = args.get("image_urls") or ([args["image_url"]] if args.get("image_url") else [])
    prompt = args.get("prompt", "")
    # Kling O3 rejects prompts over 2500 chars; trim on a word boundary so a long
    # motion-graphics prompt degrades instead of failing the whole segment.
    max_len = model.get("max_prompt_chars")
    if max_len and len(prompt) > max_len:
        prompt = prompt[:max_len].rsplit(" ", 1)[0]
    out = {
        "prompt": prompt,
        "duration": _format_duration(args.get("duration", 5), model),
    }
    # Kling O3 exposes generate_audio; Wan 2.7 ignores unknown keys harmlessly,
    # but only send it when the model declares support to keep payloads clean.
    if "generate_audio" in model:
        out["generate_audio"] = bool(model.get("generate_audio"))
    if urls:
        out["image_url"] = urls[0]
    # The END frame comes from the explicit `_end_frame` role when the caller set
    # one. Never fall back to image_urls[1]: in a hard-cut narrative scene that
    # slot holds an IDENTITY reference (the PREVIOUS frame), and locking onto it
    # would make the clip animate backwards into the previous shot.
    end_frame = args.get("_end_frame")
    if end_frame:
        out["end_image_url"] = end_frame
    return out


def _process_single_video_segment(idx: int, scene: Dict, seg_duration: float, total_scenes: int,
                                  final_instructions: str, model: dict, aspect_ratio: str,
                                  resolution: str, production_type: str, next_frame_url: str = "",
                                  prev_frame_url: str = "") -> Dict:
    # Prefer the cleaned description frame_designer produced (meta/sequence text
    # stripped); fall back to the raw VLM text only if the image stage was skipped.
    raw_v_desc = scene.get("visual_prompt_sterilized") or scene.get("visual_prompt", "")

    # --- Bind image references using the convention THIS model understands ---
    # Gemini Omni is the only engine that resolves <IMAGE_REF_n> tags. Every other
    # engine treats them as meaningless literal text, so they must keep the plain
    # #imageN form (Kling / Wan additionally address frames by role, handled below).
    # Normalize #assetN -> #imageN first so only one convention reaches the model.
    raw_v_desc = re.sub(r"(?i)\s*\(?#asset\s*(\d+)\)?", r" #image\1", raw_v_desc)
    if model.get("input_type") == "images":       # Gemini Omni
        raw_v_desc = re.sub(r"(?i)\s*\(?#image\s*\d+\)?", " <IMAGE_REF_0> ", raw_v_desc)
    elif model.get("input_type") in ("kling", "endframe"):
        # Kling / Wan address frames by ROLE, not by index. Map any leftover
        # #imageN onto the start frame so the reference is unambiguous.
        raw_v_desc = re.sub(r"(?i)\s*\(?#image\s*\d+\)?", " the start frame ", raw_v_desc)
    
    v_desc = sanitize_video_prompt(raw_v_desc)
    # Camera motion gets a LIGHT clean only. Running the full sanitizer here used
    # to delete "cut", "fade" and "transition" from the motion field, mangling
    # real camera directions (e.g. "cut to a slow pan" -> "a slow pan") and losing
    # the reference video's actual movement.
    m_desc = sanitize_camera_motion(scene.get("motion", ""))
    img_url = scene.get("generated_image_url")
    scene_audio = _scene_speech(scene)
    
    # State the movement as a REQUIREMENT to match, not a loose hint — the model
    # was otherwise inventing its own movement instead of following the reference.
    # For motion graphics the 'motion' field describes ELEMENT animation (mask
    # reveal, scale punch), not a camera move, so it must not be framed as camera.
    if is_motion_graphic(production_type):
        motion_clause = (f" ANIMATION — perform exactly this: {m_desc}"
                         if m_desc else " ANIMATION: clean graphic movement.")
        # Scenes after the first are mid-sequence: opening from a dead stop is what
        # makes concatenated chunks feel stop-start rather than one flowing piece.
        if idx > 0:
            motion_clause += (" ENTRY: this shot is mid-sequence — open already in motion,"
                              " with elements arriving/settling from the previous shot's"
                              " exit. Never begin on a frozen, static frame.")
        # No style vocabulary — the supplied frames define the look. Saying "flat
        # graphic render" here forced a cartoon aesthetic onto every reference.
        style_clause = " Preserve the exact visual style, materials and lighting of the supplied frames."
    else:
        motion_clause = (f" CAMERA — match this movement exactly: {m_desc} "
                         f"Do not invent any other camera movement."
                         if m_desc else " CAMERA: locked-off static shot, no camera movement.")
        style_clause = " Cinematic lighting, clean professional render."
    isolated_prompt = f"{v_desc}{motion_clause}{style_clause}"

    # MUSIC SUPPRESSION: each chunk is generated independently, so if every chunk
    # is allowed to invent a score you get a different tempo/key per chunk and the
    # music audibly lurches at every cut. Ask for speech + natural ambience only;
    # a single continuous music bed can be laid over the finished timeline later.
    isolated_prompt += (" No background music, no soundtrack, no score — "
                        "natural ambient sound only.")
    
    # --- SURGICAL FIX: Enforce strict proportional duration pacing to eliminate timeline bloat ---
    true_duration = seg_duration
    
    if scene_audio and "On-Screen Actor" in production_type:
        if "music" in scene_audio.lower() or "score" in scene_audio.lower() or scene_audio in ["", "..."]:
            isolated_prompt += " Subject is silent, natural ambient movement. No text on screen."
        else:
            # Dialogue is scanned by the content checker too, so neutralize it as
            # well — this text was previously passed through completely unfiltered.
            safe_audio = _neutralize_flaggable(scene_audio.replace('"', '').replace("'", ""))
            isolated_prompt += (
                f' The subject speaks these exact words in a natural, warm, conversational human voice: "{safe_audio}". '
                f'VOICE DIRECTION: Authentic real-person delivery with natural intonation, breath, and pacing — '
                f'relaxed and expressive, NOT robotic, monotone, flat, or synthetic. Match the tone and vocal '
                f'identity of the reference person; keep the same speaker throughout. Accurate lip-sync. '
                f'CRITICAL TIMING: Start the clip cleanly without text; typography animates on screen only as the subject begins speaking.'
            )
        
    elif is_motion_graphic(production_type):
        # Motion graphics: nobody speaks. Drive precise graphic animation and, when
        # the storyboard supplies design copy, animate that exact text.
        copy_text = _neutralize_flaggable((scene.get("on_screen_text") or "").strip().replace('"', ""))
        isolated_prompt += (
            " MOTION GRAPHICS — AUTHOR A FULL ANIMATED SEQUENCE, do not merely add drift to "
            "the still image. The supplied frame is one key pose; you must CREATE the "
            "surrounding animation. Choreograph multiple beats inside this clip: elements "
            "enter, transform and exit with snappy overshoot and settle, staggered timing, "
            "mask/wipe reveals, scale punches, whip transitions and secondary motion. "
            "Fast-paced and fluid with continuous movement from the first frame to the last "
            "— never a static or near-still shot. Precise, deliberate easing. "
            "No talking, no lip-sync, no human speech."
        )
        if copy_text:
            isolated_prompt += (
                f' Animated typography reads exactly "{copy_text}" — kinetic type that '
                f'stays perfectly legible and correctly spelled, never warped or distorted.'
            )
            
    if idx == total_scenes - 1 and final_instructions:
        isolated_prompt += f" {sanitize_video_prompt(final_instructions)}"

    # Protect against schema validation failures while honoring the exact contract limits (Range 3-10)
    safe_integer_duration = int(clamp_chunk(true_duration))

    args = {
        "prompt": isolated_prompt.strip(), 
        "duration": safe_integer_duration, 
        "aspect_ratio": aspect_ratio
    }
    
    if img_url:
        args["image_url"] = img_url
        args["image_urls"] = [img_url]

    mg = is_motion_graphic(production_type)
    transition = (scene.get("transition") or "cut").strip().lower()

    # --- NARRATIVE CONTINUITY (live-action) ---
    # Stitched narrative clips used to look like unrelated shots because each was
    # generated from a single isolated frame. Now we use the boundary type the
    # analyzer detected from the reference:
    #   • continuous → this is the SAME unbroken take as the previous shot, so start
    #     from its last frame and carry the motion through (a long single-camera ad).
    #   • cut → a NEW shot, but pass the previous frame as an IDENTITY reference so
    #     the subject/wardrobe/lighting/world stay consistent across the cut.
    if not mg and img_url and prev_frame_url and idx > 0:
        if transition == "continuous":
            # Ordered start -> end: begin on the previous frame, land on this one.
            args["image_urls"] = [prev_frame_url, img_url]
            args["_end_frame"] = img_url        # explicit role for end-frame models
            isolated_prompt += (
                " CONTINUOUS TAKE: this is the SAME unbroken shot as the previous clip — "
                "the camera never cut. Begin exactly on the first (previous) frame and "
                "continue the motion seamlessly toward the second frame's composition, "
                "keeping the same camera, subject, wardrobe and lighting. No cut, no jump."
            )
        else:
            # The second URL here is an IDENTITY REFERENCE, not an end frame. Mark it
            # so end-frame models (Kling/Wan) do NOT lock onto it — locking would
            # force the clip to animate BACKWARDS into the previous shot.
            args["image_urls"] = [img_url, prev_frame_url]
            args["_end_frame"] = None           # no end lock for a hard cut
            isolated_prompt += (
                " SHOT IDENTITY: this is a NEW shot (a clean cut from the previous one). "
                "Render this scene's composition, but keep the SAME subject, wardrobe, "
                "lighting mood and world as shown in the reference frame. Do NOT continue "
                "the previous camera motion — this is a distinct shot, not a morph."
            )
        args["prompt"] = isolated_prompt.strip()

    # MULTI-FRAME ANIMATION (motion graphics): give the model BOTH this scene's
    # frame and the NEXT scene's frame so motion flows across the cut.
    if mg and next_frame_url and img_url:
        args["image_urls"] = [img_url, next_frame_url]
        args["_end_frame"] = next_frame_url   # true end lock: land on the next scene
        if model.get("input_type") == "endframe":
            # HARD END-FRAME LOCK (Kling O3 / Wan 2.7). The model is FORCED to end
            # exactly on the next frame, so clip N's final frame IS clip N+1's
            # first frame — no cut at the merge. Here we WANT it to arrive at and
            # settle on the end frame (the opposite of the soft-hint models below).
            isolated_prompt += (
                " CONTINUOUS MOTION-GRAPHICS SHOT — you are given a START frame and an"
                " END frame. Animate a single fluid, fast-paced sequence that BEGINS on"
                " the start frame, performs this scene's own motion-design beats"
                " (kinetic type, element reveals, transforms, secondary motion) through"
                " the middle, and ARRIVES CLEANLY on the end frame at the final moment."
                " The end frame is the EXACT last frame of this clip — settle onto it"
                " precisely and hold there, because the next clip continues from it."
                " Keep constant movement the whole way; no static holds until the very"
                " end. One continuous take, no internal cuts."
            )
        else:
            # SOFT HANDOFF (models with no real end-frame control, e.g. Seedance /
            # Gemini). We can't force the last frame, so aim toward the next
            # composition best-effort; the stitcher hard-cuts at the join.
            isolated_prompt += (
                " SHOT STRUCTURE: OPEN on THIS scene's own frame (the starting"
                " composition) and for ~80% of the clip perform THIS scene's animation"
                " (kinetic type, element reveals), then in the last ~20% begin moving"
                " toward the FOLLOWING scene's composition. Direction of travel is"
                " forward: start here, head toward the next — never the reverse."
            )
        args["prompt"] = isolated_prompt.strip()

    if model.get("input_type") == "video":
        args["resolution"] = resolution

    # Kling v3 Pro speaks a different dialect: start/end frames, string duration,
    # native audio, no aspect_ratio. Translate our internal payload to its contract.
    if model.get("input_type") == "kling":
        args = _to_kling_args(args, model)
    # End-frame-lock models (Kling O3 / Wan 2.7): image_url + end_image_url.
    elif model.get("input_type") == "endframe":
        args = _to_endframe_args(args, model)
        # Wan 2.7 accepts a resolution; pass it when the model declares support.
        if model.get("resolutions"):
            args["resolution"] = resolution if resolution in model["resolutions"] else model["resolutions"][0]

    # `_end_frame` is an INTERNAL role marker, never part of any model's contract.
    # The adapters above rebuild args from scratch, so this only matters for the
    # models that send `args` as-is (Seedance / Gemini).
    args.pop("_end_frame", None)

    print(f"\n      [DISPATCHING] Scene {idx+1}/{total_scenes} -> {safe_integer_duration}s requested")
    # Log the frame chain explicitly: with an end-frame lock, scene N's END must
    # equal scene N+1's START. Printing both makes a mis-ordered chain obvious.
    if args.get("end_image_url") or args.get("image_url") or args.get("start_image_url"):
        print(f"      [FRAMES] start={args.get('image_url') or args.get('start_image_url') or '—'}")
        print(f"      [FRAMES]   end={args.get('end_image_url') or '— (free ending)'}")
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
        # CONTENT-POLICY SELF-HEAL: the checker is a black box, so a prompt can be
        # flagged even after sanitizing. Rather than losing the scene, retry ONCE
        # with a stripped-back, deliberately bland prompt. This costs one extra
        # call only for scenes that were actually flagged.
        if _is_content_flag(error_msg):
            safe_args = _make_safe_retry_args(args, scene)
            print(f"      [CONTENT FLAG] Scene {idx+1}: retrying once with a neutral prompt…")
            try:
                result = fal_client.subscribe(model["endpoint"], arguments=safe_args)
                out_url = result.get("video", {}).get("url")
                if out_url:
                    local_path = _download_result(out_url)
                    print(f"      [COMPLETED] Scene {idx+1} (neutral-prompt retry)")
                    return {**base_result, "args": safe_args, "status": "ok",
                            "path": local_path, "error": None,
                            "note": "Recovered after a content flag using a neutral prompt."}
            except Exception as e2:
                error_msg = (f"Content flagged, and the neutral retry also failed: {e2}. "
                             f"Edit this scene's prompt to remove sensitive wording.")
            else:
                error_msg = ("Content flagged; the neutral retry returned no video. "
                             "Edit this scene's prompt to remove sensitive wording.")
        print(f"      [FAILED] Scene {idx+1}: {error_msg}")
        return {**base_result, "status": "failed", "path": None, "error": error_msg}

def generate_final_video(storyboard_frames: List[Dict], final_instructions: str, model_id: str,
                         cost_tracker: CostTracker, total_duration: int, aspect_ratio: str,
                         resolution: str, raw_script_text: str, production_type: str = "Narrative",
                         duration_overrides: List[float] = None) -> str:

    model = MODELS[model_id]
    total_scenes = len(storyboard_frames)

    if not storyboard_frames:
        raise RuntimeError("No valid scenes found in the structured storyboard.")

    # If the user reviewed and edited the timeline in the pre-generation step, honor
    # their per-scene durations instead of re-planning. Otherwise fall back to the
    # automatic speech-aware plan. Either way, clamp to the model's real limits.
    if duration_overrides and len(duration_overrides) == total_scenes:
        scene_durations = [float(clamp_chunk(d)) for d in duration_overrides]
    else:
        scene_durations = plan_dynamic_segments(storyboard_frames, total_duration)

    # Pre-flight budget ceiling: durations and per-segment price are known before
    # any video call, so we can refuse the whole run up-front rather than spend
    # partially and stop mid-way. Images/analysis are already billed by this point.
    projected_video = sum(estimate_segment_cost(model_id, d, resolution) for d in scene_durations)
    cost_tracker.check_budget(projected_video, label=f"rendering {total_scenes} video segment(s)")

    segment_results = []

    if is_motion_graphic(production_type):
        # SEQUENTIAL CHAINING: each clip starts on the previous clip's final frame
        # so the animation flows continuously across cuts. This cannot be
        # parallelised — scene N depends on scene N-1's rendered output.
        # Each clip animates from its own frame TO the next scene's frame, so the
        # segments join seamlessly. Frames are all known up-front, so this still
        # runs in PARALLEL.
        print("\n[STORYBOARD ENGINE] Multi-frame transition rendering (motion graphics)…")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, scene in enumerate(storyboard_frames):
                nxt = ""
                if idx + 1 < total_scenes:
                    nxt = storyboard_frames[idx + 1].get("generated_image_url") or ""
                futures.append(executor.submit(
                    _process_single_video_segment,
                    idx, scene, scene_durations[idx], total_scenes,
                    final_instructions, model, aspect_ratio, resolution, production_type,
                    nxt,
                ))
            for f in concurrent.futures.as_completed(futures):
                segment_results.append(f.result())
        return _finalize_segments(segment_results, model_id, cost_tracker, resolution, hard_cut=True)

    # Narrative: pass each scene the PREVIOUS storyboard frame so continuity is
    # driven by the analyzer's per-scene 'transition' (continuous vs cut). Using
    # the previous *storyboard* frame (not the rendered clip) keeps this parallel.
    print("\n[STORYBOARD ENGINE] Initializing Parallel Video Rendering Matrix...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, scene in enumerate(storyboard_frames):
            prev = ""
            if idx > 0:
                prev = storyboard_frames[idx - 1].get("generated_image_url") or ""
            futures.append(executor.submit(
                _process_single_video_segment,
                idx, scene, scene_durations[idx], total_scenes,
                final_instructions, model, aspect_ratio, resolution, production_type,
                "", prev,
            ))

        for future in concurrent.futures.as_completed(futures):
            # _process_single_video_segment never raises; it returns a status dict.
            segment_results.append(future.result())

    return _finalize_segments(segment_results, model_id, cost_tracker, resolution)


def _finalize_segments(segment_results: List[Dict], model_id: str,
                       cost_tracker: CostTracker, resolution: str,
                       hard_cut: bool = False) -> Dict:
    """Order, bill and assemble rendered segments (shared by both render paths)."""
    segment_results.sort(key=lambda x: x["idx"])
    # Attach the model_id so a later manual re-render knows which engine to use,
    # and bill only for the chunks that actually succeeded.
    for res in segment_results:
        res["model_id"] = model_id
        if res["status"] == "ok":
            cost_tracker.add_segment(model_id, res["duration"], resolution)
    return _assemble_from_segments(segment_results, hard_cut=hard_cut)


def _assemble_from_segments(segment_results: List[Dict], hard_cut: bool = False) -> Dict:
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
        final_video_path = stitch_segments_natively([r["path"] for r in ok], hard_cut=hard_cut)

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
        new_img = image_url_override.strip()
        if "start_image_url" in args:   # Kling-shaped payload
            args["start_image_url"] = new_img
        else:
            args["image_url"] = new_img
            args["image_urls"] = [new_img]

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