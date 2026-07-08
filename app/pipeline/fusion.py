"""Reconciles the VLM breakdown with user changes into a model-appropriate final prompt."""
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
- Do NOT include any on-screen text, logos, or brand names from the breakdown.
- Output ONLY the final brief. No preamble, no labels, no explanation."""

FUSION_INSTRUCTIONS_IMAGES = """You are a video ad director working with an image-to-video generator that
receives {n} reference images, in order, tagged @Image1 through @Image{n}. You are given:
1. A "Moment 1, Moment 2..." breakdown of a reference ad, each moment matching one of these images in order.
2. Changes the user wants applied.

Write ONE final director-level prompt, 100-180 words, that:
- Explicitly references @Image1 through @Image{n} in sequence, describing the action and transition
  happening AT and BETWEEN each one, so the generator understands the full narrative arc across all images,
  not just each image in isolation.
- Applies the user's requested changes, which ALWAYS override the original mood/subject where they conflict.
- Includes a concrete audio direction line (ambient sound, music mood, or voiceover tone).
- Includes explicit photorealism qualifiers (lens type, natural motion, realistic lighting, film-like texture).
- Does NOT include any on-screen text, logos, or brand names from the breakdown.
- Output ONLY the final prompt. No preamble, no labels, no explanation."""


def fuse_prompt(breakdown: str, requested_changes: str, model_input_type: str = "video",
                 num_images: int = 0) -> str:
    """Builds the final generation prompt, branching by whether the target model uses video or image references."""
    if not requested_changes.strip():
        requested_changes = "No specific changes requested — recreate faithfully."

    user_prompt = (
        f"REFERENCE BREAKDOWN:\n{breakdown}\n\n"
        f"REQUESTED CHANGES:\n{requested_changes}"
    )

    if model_input_type == "images":
        system_prompt = FUSION_INSTRUCTIONS_IMAGES.format(n=max(num_images, 1))
    else:
        system_prompt = FUSION_INSTRUCTIONS_VIDEO

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