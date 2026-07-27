"""Analyzer — watches the reference ad and produces a CAMPAIGN BLUEPRINT:
the story spine (core idea, arc) plus per-scene entries that carry narrative
context and an adaptation brief (what to keep from the reference vs what to
replace with the user's product).

v2 changes vs v1:
  - STRICT JSON output parsed by pydantic (v1's labeled-text + regex parsing
    broke repeatedly).
  - Each scene records role_in_story + what_happens, and we derive
    previous_summary chains so the video stage never loses the thread.
  - The adaptation brief exists at all (v1 had no concept of keep-vs-replace,
    which is why outputs stayed about the reference's product).
"""
import json
import re
from typing import List, Optional

import fal_client
from pydantic import BaseModel, Field, ValidationError

VLM_ENDPOINT = "openrouter/router/video"
VLM_MODEL = "google/gemini-3.1-pro-preview"


class Adapt(BaseModel):
    keep: str = "composition, camera angle, lighting, colour grading, mood"
    replace: str = "the product, brand and any people"


class Scene(BaseModel):
    scene_number: int
    timestamp: str = ""
    keyframe_time: str = ""
    role_in_story: str = ""          # hook / problem / reveal / proof / CTA ...
    what_happens: str = ""           # plain-language beat, feeds previous_summary
    visual: str = ""                 # what the frame shows (user's product terms)
    motion: str = ""                 # camera or element movement
    dialogue: str = ""               # spoken words only, never music notes
    on_screen_text: str = ""         # short design copy, if any
    transition: str = "cut"          # cut | continuous
    assigned_assets: List[int] = Field(default_factory=list)   # 1-based
    adapt: Adapt = Field(default_factory=Adapt)


class Campaign(BaseModel):
    core_idea: str = ""
    product: str = ""
    tone: str = ""
    visual_style: str = ""
    is_motion_graphic: bool = False
    scenes: List[Scene]


def is_motion_graphic(production_type: str) -> bool:
    return "motion graphic" in (production_type or "").lower()


_PROMPT = """You are a senior commercial director. Watch the ENTIRE reference ad, then design
a {duration}-second adaptation of it for the user's product.

USER'S PRODUCT / DIRECTIVE: {directive}
USER ASSETS (referenced 1-based as they will be numbered): {assets}
FORMAT: {production_type}

Decompose into {nscenes} scenes. CRITICAL RULES:
- 'visual' must describe the USER'S product in the frame, not the reference's.
  The reference contributes style only; record what to preserve in adapt.keep
  and what to swap in adapt.replace, per scene.
- 'what_happens' is the story beat in one sentence — these chain into the ad's
  narrative, so together they must tell one complete story that lands the core idea.
- 'dialogue' contains ONLY spoken words. Music/SFX never go in dialogue.
- 'transition' is "continuous" only when the camera does not cut between scenes.
- Refer to user assets as "Asset N" (1-based) inside visual descriptions and list
  their numbers in assigned_assets.
{mg_rules}
Return STRICT JSON only, matching:
{{"core_idea": str, "product": str, "tone": str, "visual_style": str,
  "is_motion_graphic": bool, "scenes": [{{"scene_number": int, "timestamp": "M:SS-M:SS",
  "keyframe_time": "M:SS", "role_in_story": str, "what_happens": str, "visual": str,
  "motion": str, "dialogue": str, "on_screen_text": str, "transition": "cut|continuous",
  "assigned_assets": [int], "adapt": {{"keep": str, "replace": str}}}}]}}"""

_MG_RULES = """- MOTION GRAPHICS: dialogue is "" everywhere. Put punchy copy (max 6 words) in
  on_screen_text. 'visual' describes a DESIGNED composition matching the reference's
  actual rendering style. 'motion' uses motion-design vocabulary (mask reveal,
  scale punch, kinetic type stagger), never camera verbs."""


def _extract_json(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("VLM returned no JSON object")
    return json.loads(txt[start:end + 1])


def analyze_reference(video_url: str, directive: str, target_duration: int,
                      production_type: str, asset_manifest: str = "",
                      exact_scenes: Optional[int] = None) -> Campaign:
    """One VLM call -> validated Campaign. Raises with a clear message on failure."""
    nscenes = exact_scenes or max(2, min(8, round(target_duration / 5)))
    mg = is_motion_graphic(production_type)
    prompt = _PROMPT.format(
        duration=target_duration,
        directive=directive or "Direct adaptation for the user's product",
        assets=asset_manifest or "none uploaded",
        production_type=production_type,
        nscenes=nscenes,
        mg_rules=_MG_RULES if mg else "",
    )
    # Schema-verified contract: the router wants video_urls as a LIST, and
    # gemini-3.1-pro requires reasoning=true ("mandatory ... cannot be disabled").
    result = fal_client.subscribe(VLM_ENDPOINT, arguments={
        "video_urls": [video_url], "prompt": prompt, "model": VLM_MODEL,
        "reasoning": True, "temperature": 0.4,
    })
    raw = result.get("output", "")
    try:
        campaign = Campaign.model_validate(_extract_json(raw))
    except (ValueError, ValidationError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Blueprint parsing failed: {e}. Raw head: {raw[:300]}")

    campaign.is_motion_graphic = campaign.is_motion_graphic or mg
    campaign.scenes.sort(key=lambda s: s.scene_number)
    # Chain the narrative: each scene knows what the viewer just saw.
    running = []
    for scene in campaign.scenes:
        setattr(scene, "_prev", " ".join(running[-2:]))
        if scene.what_happens:
            running.append(scene.what_happens)
    return campaign


def scene_to_dict(scene: Scene, campaign: Campaign) -> dict:
    d = scene.model_dump()
    d["previous_summary"] = getattr(scene, "_prev", "")
    d["visual_prompt"] = d["visual"]            # v1-compatible alias
    d["spoken_dialogue"] = d["dialogue"]
    return d
