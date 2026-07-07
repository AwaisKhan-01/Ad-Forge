"""Reconciles Nemotron's breakdown with the user's requested changes into one coherent Seedance prompt."""
import fal_client

FUSION_MODEL = "google/gemini-2.5-flash"

FUSION_INSTRUCTIONS = """You are a video ad creative director. You are given:
1. A technical breakdown of a reference ad (shots, camera, lighting, pacing, mood).
2. Changes the user wants applied.

Write ONE final creative brief for an AI video generator, 50-150 words, following:
Subject + Action + Camera + Scene/Lighting + Style.

Rules:
- The user's requested changes ALWAYS override the original breakdown's mood/tone/subject where they conflict.
- Preserve structural details (shot count, camera movement, pacing) from the breakdown UNLESS the user's change contradicts them.
- Output ONLY the final brief. No preamble, no labels, no explanation."""


def fuse_prompt(breakdown: str, requested_changes: str) -> str:
    """Combines breakdown + user changes into a single reconciled Seedance-ready prompt."""
    if not requested_changes.strip():
        requested_changes = "No specific changes requested — recreate faithfully."

    user_prompt = (
        f"REFERENCE BREAKDOWN:\n{breakdown}\n\n"
        f"REQUESTED CHANGES:\n{requested_changes}"
    )

    result = fal_client.subscribe(
        "fal-ai/any-llm",
        arguments={
            "model": FUSION_MODEL,
            "system_prompt": FUSION_INSTRUCTIONS,
            "prompt": user_prompt,
        },
    )

    output = result.get("output")
    if not output:
        raise RuntimeError("Fusion step returned no output.")

    return output.strip()