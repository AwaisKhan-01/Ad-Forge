"""Model registry — every contract below is VERIFIED against the endpoint's live
OpenAPI schema (https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=...).

Never assume a field's type: these five video models use FOUR different duration
formats. `dialect` selects the prompt style, `payload` selects the request
adapter in generator.py. Older models are deliberately absent.
"""

# ---------------------------------------------------------------- video models
VIDEO_MODELS = {
    "seedance-2.0-fast": {
        "label": "Seedance 2.0 Fast (multi-reference)",
        "endpoint": "bytedance/seedance-2.0/fast/reference-to-video",
        "dialect": "seedance",          # concise directives, #imageN tags
        "payload": "multi_image",       # image_urls list
        "duration": {"type": "string", "min": 4, "max": 15, "auto": True},
        "aspect_ratios": ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
        "resolutions": ["480p", "720p"],
        "max_images": 9,
        "audio": True,                  # generate_audio flag
        "price_per_sec": 0.2419 * 0.6,
    },
    "gemini-omni-flash": {
        "label": "Gemini Omni Flash (multi-reference)",
        "endpoint": "google/gemini-omni-flash/reference-to-video",
        "dialect": "gemini",            # rich paragraph, <IMAGE_REF_n> tags
        "payload": "multi_image",
        "duration": {"type": "integer", "min": 3, "max": 10},
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": None,
        "max_images": 10,
        "audio": True,                  # speech baked into generation
        "price_per_sec": 0.10,
    },
    "kling-o3": {
        "label": "Kling O3 (end-frame lock · best value)",
        "endpoint": "fal-ai/kling-video/o3/standard/image-to-video",
        "dialect": "kling",             # action-focused, start/end frame roles
        "payload": "endframe",          # image_url + end_image_url
        "duration": {"type": "string", "min": 3, "max": 15},
        "aspect_ratios": ["auto"],      # inferred from start image
        "resolutions": None,
        "max_images": 2,
        "audio": True,                  # generate_audio (we default it off)
        "generate_audio_default": False,
        "max_prompt_chars": 2500,       # schema: prompt maxLength
        "price_per_sec": 0.084,         # audio off
    },
    "wan-v2.7": {
        "label": "Wan 2.7 (end-frame lock)",
        "endpoint": "fal-ai/wan/v2.7/image-to-video",
        "dialect": "wan",               # plain descriptive, motion-forward
        "payload": "endframe",
        "duration": {"type": "integer", "min": 2, "max": 15},
        "aspect_ratios": ["auto"],
        "resolutions": ["720p", "1080p"],
        "max_images": 2,
        "audio": False,                 # no generate_audio in schema
        "price_per_sec": 0.10,          # 720p; 1080p = 0.15
    },
    "veo-3.1-flf": {
        "label": "Veo 3.1 (first+last frame · premium)",
        "endpoint": "fal-ai/veo3.1/first-last-frame-to-video",
        "dialect": "veo",               # cinematic prose, audio cues inline
        "payload": "flf",               # first_frame_url + last_frame_url (BOTH required)
        "duration": {"type": "suffixed", "values": [4, 6, 8]},   # '4s' | '6s' | '8s'
        "aspect_ratios": ["auto", "16:9", "9:16"],
        "resolutions": ["720p", "1080p"],
        "max_images": 2,
        "audio": True,
        "price_per_sec": 0.40,          # with audio, 720/1080p
    },
}

DEFAULT_VIDEO_MODEL = "seedance-2.0-fast"

# ---------------------------------------------------------------- image models
IMAGE_MODELS = {
    "nano-banana-2-edit": {
        "label": "Nano Banana 2 (edit)",
        "endpoint": "fal-ai/nano-banana-2/edit",
        "supports_size": False,
        "flat_price": 0.015,
        # verified schema: resolution enum, default '1K'. Was not exposed at all.
        "resolutions": ["0.5K", "1K", "2K", "4K"],
        "default_resolution": "1K",
    },
    "gpt-image-2": {
        "label": "GPT Image 2 (edit)",
        "endpoint": "fal-ai/gpt-image-2/edit",
        "supports_size": True,
        # verified schema: image_size is an ENUM of these named presets, not a
        # pixel-dimension string. The old list ("1024x1024" etc.) matched NO
        # valid enum value, so fal silently fell back on every call — which is
        # exactly why only 9:16/16:9 ever came out regardless of selection.
        "sizes": ["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9",
                 "landscape_4_3", "landscape_16_9"],
        "qualities": ["auto", "low", "medium", "high"],
        # fal's OpenAPI schema doesn't publish $ pricing; these are OpenAI's
        # confirmed GPT-Image tier prices (square vs non-square), remapped onto
        # the real size enum above. 'auto' uses the non-square tier as a safe
        # (slightly conservative) budget estimate.
        "pricing": {
            ("square_hd", "low"): 0.011, ("square_hd", "medium"): 0.042, ("square_hd", "high"): 0.167,
            ("square", "low"): 0.011, ("square", "medium"): 0.042, ("square", "high"): 0.167,
            ("portrait_4_3", "low"): 0.016, ("portrait_4_3", "medium"): 0.063, ("portrait_4_3", "high"): 0.25,
            ("portrait_16_9", "low"): 0.016, ("portrait_16_9", "medium"): 0.063, ("portrait_16_9", "high"): 0.25,
            ("landscape_4_3", "low"): 0.016, ("landscape_4_3", "medium"): 0.063, ("landscape_4_3", "high"): 0.25,
            ("landscape_16_9", "low"): 0.016, ("landscape_16_9", "medium"): 0.063, ("landscape_16_9", "high"): 0.25,
            ("auto", "low"): 0.016, ("auto", "medium"): 0.063, ("auto", "high"): 0.25, ("auto", "auto"): 0.063,
        },
    },
}

DEFAULT_IMAGE_MODEL = "nano-banana-2-edit"


def image_price(model_key: str, size: str = "", quality: str = "") -> float:
    m = IMAGE_MODELS.get(model_key, IMAGE_MODELS[DEFAULT_IMAGE_MODEL])
    if not m["supports_size"]:
        return m.get("flat_price", 0.015)
    pricing = m.get("pricing", {})
    return pricing.get((size, quality)) or pricing.get((size, "medium")) or 0.05


def video_price(model_key: str, seconds: float, resolution: str = "720p") -> float:
    m = VIDEO_MODELS.get(model_key)
    if not m:
        return 0.0
    rate = m["price_per_sec"]
    if model_key == "wan-v2.7" and resolution == "1080p":
        rate = 0.15
    return seconds * rate


def clamp_duration(model_key: str, seconds: float):
    """Clamp to the model's schema range and serialize in ITS format.

    seedance: string, supports 'auto' · gemini: int · kling: string ·
    wan: int · veo: '4s'/'6s'/'8s'. Verified per-endpoint; do not unify.
    """
    m = VIDEO_MODELS[model_key]
    spec = m["duration"]
    if spec["type"] == "suffixed":
        best = min(spec["values"], key=lambda v: abs(v - seconds))
        return f"{best}s"
    val = int(round(max(spec["min"], min(spec["max"], seconds))))
    return str(val) if spec["type"] == "string" else val
