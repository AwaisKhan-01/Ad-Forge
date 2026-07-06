import os
from dotenv import load_dotenv

load_dotenv()

FAL_KEY = os.environ.get("FAL_KEY")

if not FAL_KEY:
    raise RuntimeError(
        "FAL_KEY not set. Each user must supply their own key via .env "
        "or `docker run -e FAL_KEY=...`. See docs/SETUP.md."
    )

# Model endpoints (pin exact model paths so upgrades are deliberate, not silent)
NEMOTRON_MODEL = "nvidia/nemotron-3-nano-omni/video"
SEEDANCE_MODEL = "bytedance/seedance-2.0/fast/reference-to-video"

# Limits
MAX_VIDEO_DURATION_SEC = 120  # Nemotron's hard cap
OUTPUT_DIR = "outputs"
