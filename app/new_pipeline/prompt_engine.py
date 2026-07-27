"""Prompt engine — the ONLY place prompts are assembled.

Fixes the two v1 failure modes:
  1. One hardcoded prompt for every model. Each model reads a different dialect;
     v1 sent Gemini-style prose to all of them.
  2. Contradictory boilerplate ("no text on screen" + "animated typography
     reads ...") stacked by independent code paths. Here every prompt is built
     from mutually-exclusive blocks chosen ONCE, so contradictions are
     structurally impossible.

Image notation: #image1..#imageN for every model EXCEPT Gemini Omni, which
resolves <IMAGE_REF_n> tags (n is 0-based). Nothing else — @image is obsolete.
"""
import re

from app.new_pipeline.model_registry import VIDEO_MODELS

# Words that commonly trip content classifiers on innocent ad copy.
_FLAG_PHRASES = [
    (r"\bshoot(?:ing|s)?\s+(?:him|her|them|people|someone|each other)\b", "confronting"),
    (r"\bgun\s*shot\b", "loud bang"),
    (r"\bblow(?:ing|s)?\s+up\b", "bursting"),
]
_FLAG_WORDS = {
    "kill": "stop", "blood": "red liquid", "gun": "prop", "weapon": "prop",
    "knife": "blade tool", "bomb": "burst", "explosion": "burst of light",
    "naked": "bare", "nude": "bare", "drug": "supplement", "violence": "intensity",
    "suicide": "crisis", "ripped": "distressed",
}


def neutralize(text: str) -> str:
    """Soften classifier-bait wording without touching film language."""
    if not text:
        return ""
    out = text
    for pat, repl in _FLAG_PHRASES:
        out = re.sub(rf"(?i){pat}", repl, out)
    for bad, safe in _FLAG_WORDS.items():
        out = re.sub(rf"(?i)\b{re.escape(bad)}\b", safe, out)
    return out


def to_model_notation(text: str, model_key: str) -> str:
    """Translate #imageN tags into the notation THIS model resolves."""
    dialect = VIDEO_MODELS.get(model_key, {}).get("dialect", "")
    text = re.sub(r"(?i)#asset\s*(\d+)", r"#image\1", text or "")
    if dialect == "gemini":  # 1-based tags -> 0-based IMAGE_REF
        return re.sub(r"(?i)#image\s*(\d+)",
                      lambda m: f"<IMAGE_REF_{int(m.group(1)) - 1}>", text)
    return re.sub(r"(?i)#image\s*(\d+)", r"#image\1", text)


def _story_context(scene: dict, campaign: dict) -> str:
    """The narrative thread v1 dropped: what the ad is about, what just
    happened, and this shot's specific job in the arc."""
    parts = []
    idea = (campaign or {}).get("core_idea", "")
    if idea:
        parts.append(f"This ad's core idea: {idea}.")
    prev = scene.get("previous_summary", "")
    if prev:
        parts.append(f"What just happened: {prev}.")
    job = scene.get("role_in_story", "")
    if job:
        parts.append(f"This shot's job: {job}.")
    return " ".join(parts)


def _text_block(scene: dict, text_mode: str) -> str:
    """EXACTLY one text policy per scene — never two."""
    copy = neutralize((scene.get("on_screen_text") or "").strip())
    if text_mode == "keep":
        return ("The frame already contains finished lettering. Animate the EXISTING "
                "text as-is (reveal, mask, scale) — same wording, font, colour. "
                "Add no new text.")
    if text_mode == "none":
        return "No text, captions or watermarks anywhere in the frame."
    if copy:
        return (f'Animated typography reads exactly "{copy}" — legible, correctly '
                f"spelled, never warped.")
    return ""


def _speech_block(scene: dict, model: dict) -> str:
    """Speech only for models that generate audio; silence directive otherwise."""
    line = neutralize((scene.get("dialogue") or "").strip())
    if not model.get("audio"):
        return ""                       # silent model: saying anything is noise
    if not line:
        return "No speech. Natural ambient sound only; no music (added in post)."
    tone = scene.get("delivery_tone", "warm, conversational")
    return (f'The subject speaks, {tone}, with natural pacing and accurate lip-sync: '
            f'"{line}". No background music (added in post).')


def build_video_prompt(scene: dict, campaign: dict, model_key: str,
                       text_mode: str = "add", motion_graphic: bool = False) -> str:
    """Assemble the per-scene, per-model prompt from exclusive blocks."""
    model = VIDEO_MODELS[model_key]
    dialect = model["dialect"]

    visual = neutralize(scene.get("visual", "") or scene.get("visual_prompt", ""))
    motion = neutralize(scene.get("motion", ""))
    blocks = [_story_context(scene, campaign), visual]

    if motion_graphic:
        blocks.append(
            "Motion graphics: author the full animation — staggered element reveals, "
            "kinetic type, overshoot-and-settle easing, constant fluid movement. "
            "No people, no speech.")
        if motion:
            blocks.append(f"Choreography: {motion}.")
    else:
        if motion:
            blocks.append(f"Camera: {motion}. Match this movement exactly.")
        blocks.append(_speech_block(scene, model))

    blocks.append(_text_block(scene, text_mode))

    # Dialect shaping — order and tone differ per model family.
    if dialect == "gemini":
        blocks.append("Cinematic lighting, photorealistic, coherent motion throughout.")
    elif dialect == "seedance":
        blocks.insert(0, "")            # seedance prefers directives up front, no preamble
        blocks.append("High production value. One continuous shot, no internal cuts.")
    elif dialect in ("kling", "wan"):
        blocks.append("Begin exactly on the start frame."
                      + (" Arrive cleanly on the end frame and settle there — the next "
                         "clip continues from it." if scene.get("_has_end_frame") else ""))
    elif dialect == "veo":
        blocks.append("Natural sound design matching the action; smooth interpolation "
                      "from first to last frame.")

    prompt = " ".join(b for b in blocks if b).strip()
    prompt = to_model_notation(prompt, model_key)

    cap = model.get("max_prompt_chars")
    if cap and len(prompt) > cap:
        prompt = prompt[:cap].rsplit(" ", 1)[0]
    return prompt


def build_frame_prompt(scene: dict, campaign: dict, has_user_assets: bool,
                       text_mode: str = "add", font_style_desc: str = "") -> str:
    """Storyboard image prompt. THE core v1 flaw fix: the reference frame
    (#image1) contributes STYLE — composition, lighting, grading, energy — while
    the CONTENT is the user's product/brand (#image2..N). v1 lightly edited the
    reference frame, so outputs stayed about the reference's product."""
    adapt = scene.get("adapt", {}) or {}
    keep = adapt.get("keep", "composition, camera angle, lighting, colour grading, mood")
    replace = adapt.get("replace", "the product, brand and any people")
    visual = neutralize(scene.get("visual", "") or scene.get("visual_prompt", ""))
    product = (campaign or {}).get("product", "the user's product")

    if has_user_assets:
        subject = (f"The subject is {product}, exactly as shown in the supplied asset "
                   f"images — same design, colours, materials and branding.")
    else:
        subject = f"The subject is {product}, rendered faithfully from this description."

    parts = [
        f"Use #image1 only as a style reference: keep its {keep}.",
        f"Replace {replace} entirely. {subject}",
        visual,
    ]
    if text_mode == "keep":
        parts.append("Preserve all existing text and logos exactly — wording, font, "
                     "position, colour. Add no new text.")
    elif text_mode == "none":
        parts.append("No text, letters, logos or branding anywhere.")
    elif font_style_desc:
        copy = neutralize((scene.get("on_screen_text") or scene.get("dialogue") or "").strip())
        if copy:
            parts.append(f'Integrated typography reading exactly "{copy}", styled as {font_style_desc}.')
    parts.append("High production value, professional composition.")
    return " ".join(p for p in parts if p).strip()
