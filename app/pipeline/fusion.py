"""Reconciles VLM breakdown/captions with user changes into model-appropriate final prompts."""
import fal_client

FUSION_MODEL = "google/gemini-2.5-flash"

FUSION_INSTRUCTIONS_VIDEO = """You are a video ad creative director. You are given:
1. A technical breakdown of a reference ad (shots, camera, lighting, pacing, mood, audio cues).
2. Changes the user wants applied.

Write ONE final creative brief for an AI video generator, 80-150 words, covering in order:
Subject + Action + Camera + Scene/Lighting + Audio + Style.

Rules:
- The user's requested changes ALWAYS override the original breakdown's mood/tone/subject where they conflict.
- Preserve structural details (shot count, camera movement, pacing) UNLESS the user's change contradicts them.
- ALWAYS include a concrete audio direction line (ambient sound, music mood, or voiceover tone).
- ALWAYS include explicit photorealism qualifiers: lens/camera type, realistic lighting physics, film-like grain/texture.
- Explicitly state: no fade-in or fade-out, full content visible at the very first and last frame.
- Do NOT include any on-screen text, logos, or brand names from the breakdown.
- Output ONLY the final brief. No preamble, no labels, no explanation."""

FUSION_INSTRUCTIONS_IMAGES = """You are a video ad director working with an image-to-video generator that
receives {n} reference images, in order, tagged @Image1 through @Image{n}. You are given:
1. The ACTUAL factual caption of each image (this is ground truth — do not contradict it).
2. Changes the user wants applied.

Write ONE final director-level prompt, 100-180 words, that:
- References @Image1 through @Image{n} in sequence, describing ONLY what each caption actually shows,
  plus the action/transition happening between them.
- Applies the user's requested changes, which override mood/subject where they conflict with the captions.
- Includes a concrete audio direction line (ambient sound, music mood, or voiceover tone).
- Includes explicit photorealism qualifiers (lens type, natural motion, realistic lighting, film-like texture).
- Explicitly states: no fade-in or fade-out, full content visible at the very first and last frame.
- Does NOT include any on-screen text, logos, or brand names.
- Output ONLY the final prompt. No preamble, no labels, no explanation."""


def fuse_prompt(breakdown: str, requested_changes: str, model_input_type: str = "video") -> str:
    """Fusion for video-reference models (e.g. Seedance) — uses the full Nemotron breakdown."""
    if not requested_changes.strip():
        requested_changes = "No specific changes requested — recreate faithfully."

    user_prompt = f"REFERENCE BREAKDOWN:\n{breakdown}\n\nREQUESTED CHANGES:\n{requested_changes}"

    result = fal_client.subscribe(
        "fal-ai/any-llm",
        arguments={
            "model": FUSION_MODEL,
            "system_prompt": FUSION_INSTRUCTIONS_VIDEO,
            "prompt": user_prompt,
        },
    )
    output = result.get("output")
    if not output:
        raise RuntimeError("Fusion step returned no output.")
    return output.strip()


def fuse_prompt_images(image_captions: list[str], requested_changes: str) -> str:
    """Fusion for image-reference models (e.g. Omni) — grounded in real per-frame captions."""
    if not requested_changes.strip():
        requested_changes = "No specific changes requested — recreate faithfully."

    captions_text = "\n".join(f"@Image{i+1}: {cap}" for i, cap in enumerate(image_captions))
    user_prompt = f"IMAGE CAPTIONS:\n{captions_text}\n\nREQUESTED CHANGES:\n{requested_changes}"
    system_prompt = FUSION_INSTRUCTIONS_IMAGES.format(n=len(image_captions))

    result = fal_client.subscribe(
        "fal-ai/any-llm",
        arguments={
            "model": FUSION_MODEL,
            "system_prompt": system_prompt,
            "prompt": user_prompt,
        },
    )
    output = result.get("output")
    if not output:
        raise RuntimeError("Fusion step returned no output.")
    return output.strip()