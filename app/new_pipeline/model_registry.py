MODELS = {
    "seedance-2.0-fast": {
        "label": "Seedance 2.0 (Fast)",
        "endpoint": "bytedance/seedance-2.0/fast/reference-to-video",
        "input_type": "video",
        "durations": list(range(4, 16)),   # 4-15s per segment
        "aspect_ratios": ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
        "resolutions": ["480p", "720p"],
        "max_images": 9,
    },
    "gemini-omni-flash": {
        "label": "Gemini Omni Flash",
        "endpoint": "google/gemini-omni-flash/reference-to-video",
        "input_type": "images",
        "durations": list(range(3, 11)),   # 3-10s per segment
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": None,
        "max_images": 10,                  # confirmed by real testing
    },
}