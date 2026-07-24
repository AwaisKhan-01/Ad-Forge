"""VLM-as-judge keyframe QA (flag-only).

In fast-paced reference videos, the frame extracted at a scene's keyframe_time
sometimes shows a *different* moment than the scene's visual_prompt describes
(the prompt talks about a person, the extracted frame is a UI screen, etc.).
That mismatch then feeds the image edit and the whole scene drifts.

This module asks a cheap vision model one question — does this frame match the
prompt? — and returns a verdict. It NEVER changes the pipeline's behaviour: the
verdict is attached to the scene so the UI can surface a warning, and the user
decides whether to re-pick the keyframe via the inspect panel. Auto-flag only.
"""
import json
import fal_client

# Cheap, non-deprecated vision router. Same family the analyzer already uses.
QA_ENDPOINT = "openrouter/router/vision"
QA_MODEL = "google/gemini-2.5-flash"

_SYSTEM = (
    "You are a strict storyboard QA checker. You are shown ONE frame extracted "
    "from a reference video and the text prompt that is supposed to describe that "
    "frame. Decide whether the frame's main subject and action genuinely match "
    "the prompt. Ignore small stylistic wording; only flag a MISMATCH when the "
    "frame clearly shows a different subject, scene, or moment than the prompt "
    "describes. Reply with STRICT JSON only."
)

_PROMPT_TMPL = (
    'Prompt for this scene:\n"{prompt}"\n\n'
    'Does the frame match this prompt?\n'
    'Return exactly: {{"match": true|false, "reason": "<one short sentence>"}}'
)


def _parse_verdict(raw: str) -> dict:
    """Best-effort parse of the model's JSON; degrade gracefully to 'unknown'."""
    if not raw:
        return {"status": "unknown", "reason": "empty QA response"}
    txt = raw.strip()
    # Strip a ```json fence if the model added one.
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[txt.find("{"):txt.rfind("}") + 1] if "{" in txt else txt
    else:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            txt = txt[s:e + 1]
    try:
        data = json.loads(txt)
        match = bool(data.get("match"))
        return {
            "status": "match" if match else "mismatch",
            "reason": str(data.get("reason", "")).strip(),
        }
    except Exception:
        return {"status": "unknown", "reason": "unparseable QA response"}


def check_keyframe_match(frame_url: str, visual_prompt: str) -> dict:
    """Return a verdict dict: {status: match|mismatch|unknown, reason: str}.

    Flag-only and fail-open: any error yields status 'unknown' and never raises,
    so QA can never break a storyboard run.
    """
    prompt = (visual_prompt or "").strip()
    if not frame_url or not prompt:
        return {"status": "unknown", "reason": "missing frame or prompt"}
    try:
        result = fal_client.subscribe(
            QA_ENDPOINT,
            arguments={
                "image_urls": [frame_url],
                "prompt": _PROMPT_TMPL.format(prompt=prompt.replace('"', "'")),
                "system_prompt": _SYSTEM,
                "model": QA_MODEL,
                "temperature": 0,
            },
        )
        return _parse_verdict(result.get("output", ""))
    except Exception as e:
        return {"status": "unknown", "reason": f"QA call failed: {e}"}
