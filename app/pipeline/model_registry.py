"""Central registry of supported generation models and their real constraints."""

MODELS = {
    "seedance-2.0-fast": {
        "label": "Seedance 2.0 (Fast)",
        "endpoint": "bytedance/seedance-2.0/fast/reference-to-video",
        "input_type": "video",          # takes the ad video directly
        "durations": list(range(4, 16)),  # 4-15s
        "aspect_ratios": ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
        "resolutions": ["480p", "720p"],
    },
    "gemini-omni-flash": {
        "label": "Gemini Omni Flash",
        "endpoint": "google/gemini-omni-flash/reference-to-video",
        "input_type": "images",         # needs extracted frames, not video
        "durations": list(range(3, 11)),  # 3-10s
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": None,            # no resolution param
    },
}