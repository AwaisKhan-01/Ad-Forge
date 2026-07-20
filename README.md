Ad-Forge
========

Overview
--------
Ad-Forge is an automated short-form video production pipeline designed to convert a reference video and user assets into a condensed, production-ready commercial optimized for social platforms. The system analyzes a reference video, generates a structured storyboard, creates static keyframes and edited images, produces synchronized TTS audio, renders per-scene video segments via third-party generative models, and stitches the segments into a final master file.

Key Features
------------
- Scene analysis and pacing estimation from a reference video.
- Structured storyboard generation using Vision+Language models with strict JSON schema validation.
- Keyframe extraction and image editing (typography-aware) for visual composition.
- Cloned TTS voice generation and tempo scaling to match target duration.
- Per-scene parallel rendering using configurable model backends.
- Native, lossless FFmpeg stitching of final segments.
- Cost estimation utilities for predictability of model usage.

Architecture Overview
---------------------
- `app/new_pipeline/analyzer.py`: Analyzes reference video and produces a compressed storyboard (scenes, visual prompts, motion, dialogue).
- `app/new_pipeline/frame_designer.py`: Extracts exact keyframes, compiles diffusion prompts, and invokes image-edit models for static frames and typography.
- `app/new_pipeline/generator.py`: Plans per-scene durations, dispatches generative model calls in parallel, downloads results and aggregates them.
- `app/new_pipeline/audio_engine.py`: Produces cloned TTS voiceovers and scales audio to the target runtime.
- `app/new_pipeline/stitcher.py`: Concatenates rendered segments using FFmpeg concat demuxer with stream-copy.
- `app/new_pipeline/uploader.py`: Uploads artifacts to fal.ai storage for downstream models to consume.
- `app/new_pipeline/cost_tracker.py`: Tracks estimated costs for video and LLM calls.
- `app/new_pipeline/model_registry.py`: Registry of supported model endpoints and capabilities.

Prerequisites
-------------
- Linux or macOS host with Python 3.9+.
- FFmpeg and ffprobe available on PATH.
- `yt-dlp` installed (used for downloading reference videos).
- Network access for third-party model/service APIs (fal.ai endpoints).
- Python dependencies listed in `requirements.txt`.

Installation
------------
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Configuration
-------------
Environment variables and secrets expected by the pipeline should be provided via a `.env` file or the environment. Typical variables include API keys and endpoints used by `fal_client` and other services. See `app/config.py` for configuration points.

Basic Usage
-----------
The repository exposes a programmatic interface in `app/` and an orchestrator entrypoint at `app/main.py`.

Example (conceptual):

```bash
# Download reference video (via app/downloader or externally)
# Run the orchestrator script or import pipeline modules in your own runner
python -m app.main --reference-url "<video_url>" --duration 15 --production-type "Narrative"
```

Refer to `app/main.py` for CLI flags and runtime options.

Testing and Validation
----------------------
- Unit tests and lightweight integration checks should exercise each stage individually: analyzer, frame extraction, image edits, TTS generation, segment rendering, and stitching.
- When running integration tests, stub or mock external model calls (fal.ai) to avoid cost and variability.

Operational Notes
-----------------
- The pipeline uses parallel execution for scene processing; monitor CPU, memory, and network when running multiple simultaneous segments.
- FFmpeg is used extensively—ensure the installed build supports required filters (e.g., `select`, `atempo`).
- Check `app/new_pipeline/cost_tracker.py` for cost estimates before running large batches.

Contributing
------------
- Follow the existing code style and typing patterns (Pydantic models are used in analysis). Keep changes minimal and focused.
- Add unit tests for new features and run them locally before opening pull requests.

License
-------
This repository includes a `LICENSE` file; consult it for license details.

Further Reading
---------------
- See `app/new_pipeline/` for individual component implementations and `app/main.py` for orchestration logic.

