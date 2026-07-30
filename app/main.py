"""
Gradio UI for AdForge: Enterprise Dashboard.
Features a tabbed workflow, professional CSS, decoupled logic, and dynamic API Key injection.
Located in: app/main.py
"""
import os
import gradio as gr
from app.new_pipeline.downloader import download_video
from app.new_pipeline.uploader import upload_video
from app.new_pipeline.model_registry import MODELS
from app.new_pipeline.cost_tracker import CostTracker, BudgetExceeded

from app.new_pipeline.analyzer import analyze_and_extract_scenes, is_motion_graphic
from app.new_pipeline.frame_designer import generate_storyboard_frames, _process_single_scene, parse_extracted_scenes, IMAGE_MODELS, DEFAULT_IMAGE_MODEL
from app.new_pipeline.generator import (
    generate_final_video, regenerate_single_segment,
    build_timeline_preview, clamp_chunk, MODEL_MIN_CHUNK, MODEL_MAX_CHUNK,
)
from app.new_pipeline import session_store

# Fixed pool of per-scene duration editors. A ≤60s ad never needs more scenes
# than this; unused rows stay hidden. Editing a variable number of components in
# Gradio requires a fixed pool made visible up to the real scene count.
MAX_TIMELINE_ROWS = 30

# --- SLEEK ENTERPRISE CSS ---
CUSTOM_CSS = """
.gradio-container { background-color: #0b0f19 !important; font-family: 'Inter', sans-serif; }
#header-container { text-align: center; margin-bottom: 20px; padding: 20px; background: linear-gradient(90deg, #1e1b4b 0%, #312e81 100%); border-radius: 12px; border: 1px solid #4338ca; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
h1, h2, h3, p { color: #f3f4f6 !important; }
.panel-box { border-radius: 12px !important; border: 1px solid #1f2937 !important; background-color: #111827 !important; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.gr-button-primary { background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important; border: none !important; box-shadow: 0 0 10px rgba(139, 92, 246, 0.5) !important; transition: all 0.2s ease-in-out; }
.gr-button-primary:hover { transform: translateY(-2px); box-shadow: 0 0 15px rgba(139, 92, 246, 0.8) !important; }
.gr-button-secondary { background: #374151 !important; border: 1px solid #4b5563 !important; }
.gr-button-secondary:hover { background: #4b5563 !important; }
.tabs { border-bottom: 2px solid #374151 !important; }
.tabitem { border: none !important; background: transparent !important; }
"""

# NOTE: .set() keys without the _dark suffix only style Gradio's LIGHT palette.
# Both variants are set to the same dark values so the UI is identical no
# matter which palette the visitor's browser starts in.
THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="gray").set(
    body_background_fill="#0b0f19", block_background_fill="#111827",
    block_border_color="#1f2937", button_primary_background_fill="#6366f1",
    body_background_fill_dark="#0b0f19", block_background_fill_dark="#111827",
    block_border_color_dark="#1f2937", button_primary_background_fill_dark="#6366f1",
    body_text_color="#f3f4f6", body_text_color_dark="#f3f4f6",
    block_label_text_color="#9ca3af", block_label_text_color_dark="#9ca3af",
    input_background_fill="#1f2937", input_background_fill_dark="#1f2937",
)

def set_fal_key(api_key: str):
    if not api_key or not api_key.strip():
        raise gr.Error("Error: Missing API Key. Please paste your FAL API Key in the top header.")
    os.environ["FAL_KEY"] = api_key.strip()

def _asset_manifest(uploaded_assets) -> str:
    """Numbered inventory of uploads, 1-based to match the UI and the VLM prompt."""
    if not uploaded_assets:
        return ""
    lines = []
    for i, path in enumerate(uploaded_assets, start=1):
        if path:
            lines.append(f"Asset {i}: {os.path.basename(path)}")
    return "\n".join(lines)

def render_asset_list(uploaded_assets):
    """Live 'Asset 1 — file.png' list shown under the upload box so the user knows
    exactly which number to reference in the Creative Directive."""
    manifest = _asset_manifest(uploaded_assets)
    if not manifest:
        return gr.update(value="_No assets uploaded yet. Upload files, then refer to them as **Asset 1**, **Asset 2**, … in your Creative Directive._", visible=True)
    rows = "\n".join(
        f"- **Asset {i}** — `{os.path.basename(p)}`"
        for i, p in enumerate(uploaded_assets, start=1) if p
    )
    return gr.update(value=f"**Your assets** (refer to these numbers in the Creative Directive):\n\n{rows}", visible=True)

def step1_download(url):
    """Fetch the reference from a URL (cached — only downloads once per link)."""
    if not url or not url.strip():
        raise gr.Error("Paste a reference URL, or upload/drag a local video into the Source Video box.")
    local_path = download_video(url)
    return local_path, gr.update(interactive=True)

def on_source_ready(local_video_path):
    """Enable the Analyze step whenever a source video is present — whether it was
    fetched from a URL or loaded directly from the user's machine."""
    return gr.update(interactive=bool(local_video_path))

def step2_extract_scenes(api_key, local_video_path, creative_directive, target_duration, production_type, uploaded_assets):
    set_fal_key(api_key)
    if not local_video_path:
        return "Error: Please download a video first."
    try:
        hosted_video_url = upload_video(local_video_path)
        # Give the VLM the actual 1-based inventory of uploads alongside the user's
        # description of them. Without the manifest the model has to guess how many
        # assets exist and what numbering to use, which is what caused it to emit
        # out-of-range / 0-based indices and swap the wrong asset into a scene.
        manifest = _asset_manifest(uploaded_assets)
        if manifest:
            asset_instructions = (
                f"The user uploaded {len([a for a in uploaded_assets if a])} asset(s), "
                f"numbered starting at 1:\n{manifest}\n\n"
                f"What the user says about them: {creative_directive or '(no description given)'}"
            )
        else:
            asset_instructions = ""
        extracted_text = analyze_and_extract_scenes(
            video_url=hosted_video_url,
            target_duration=int(target_duration),
            creative_directive=creative_directive,
            production_type=production_type,
            asset_instructions=asset_instructions
        )
        return extracted_text
    except Exception as e:
        return f"PIPELINE ERROR: {str(e)}"

def step3_generate_storyboard(api_key, edited_prompts, uploaded_assets, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality, production_type="", run_qa=True):
    set_fal_key(api_key)
    if not edited_prompts:
        return [], None
    asset_paths = uploaded_assets if uploaded_assets else []
    try:
        mg = is_motion_graphic(production_type)
        storyboard_data = generate_storyboard_frames(edited_prompts, asset_paths, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality, motion_graphic=mg, run_qa=run_qa)

        # Stamp the image-gen settings onto each frame so the final cost estimate
        # (step 4) can price the images with the actual model/size/quality used.
        for frame in storyboard_data:
            frame["image_model"] = image_model
            frame["font_style"] = font_style   # video stage needs it for text handling
            frame["image_size"] = image_size
            frame["image_quality"] = image_quality

        # --- SURGICAL FIX: Update Gallery Captions for Split Schema ---
        gallery_format = []
        mismatches = []
        for frame in storyboard_data:
            if frame.get("generated_image_url"):
                display_text = frame.get('spoken_dialogue') or frame.get('sfx_music', '')
                # Flag-only keyframe QA: mark the caption, don't change the render.
                qa = frame.get("qa_keyframe") or {}
                prefix = ""
                if qa.get("status") == "mismatch":
                    prefix = "⚠ "
                    mismatches.append((frame['scene_number'], qa.get('reason', '')))
                gallery_format.append((frame["generated_image_url"],
                                       f"{prefix}Scene {frame['scene_number']}: {display_text}"))

        if mismatches:
            msg = "Keyframe/prompt mismatch flagged — review & re-pick if needed:\n" + \
                  "\n".join(f"• Scene {n}: {r}" for n, r in mismatches)
            gr.Warning(msg)

        # Autosave so a server restart doesn't lose this (expensive) storyboard.
        session_store.save_session(storyboard_frames=storyboard_data)
        return gallery_format, storyboard_data
    except Exception as e:
        print(f"Storyboard compilation error: {str(e)}")
        return [], None

def _gallery_to_urls(gallery, uploader):
    """Turn a gr.Gallery value into a flat list of image URLs, uploading any
    freshly-added LOCAL files (the '+' tiles) to fal so they get a usable URL."""
    urls = []
    for item in (gallery or []):
        # gr.Gallery yields (path_or_url, caption) tuples, or bare paths/dicts.
        src = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(src, dict):
            src = src.get("path") or src.get("url") or src.get("name") or ""
        if not src:
            continue
        urls.append(src if str(src).startswith("http") else uploader(src))
    return urls

def step3_regenerate_single(api_key, edited_prompts, scene_num_to_regen, uploaded_assets, local_video_path, aspect_ratio, font_style, current_storyboard_state, image_model, image_size, image_quality, frame_prompt_override="", frame_images_gallery=None, production_type=""):
    set_fal_key(api_key)
    if not current_storyboard_state or not scene_num_to_regen:
        return gr.update(), current_storyboard_state
    try:
        fresh_scenes = parse_extracted_scenes(edited_prompts)
        target_scene_data = next((s for s in fresh_scenes if str(s["scene_number"]) == str(scene_num_to_regen)), None)

        if not target_scene_data:
            gr.Warning(f"Scene {scene_num_to_regen} not found in the text prompts panel.")
            return gr.update(), current_storyboard_state

        asset_urls = []
        if uploaded_assets:
            for asset in uploaded_assets:
                url = upload_video(asset) if not asset.startswith("http") else asset
                asset_urls.append(url)

        # The inspect gallery is the source of truth for this scene's input images.
        # Whatever the user left in it (originals + any '+'-added tiles, in order)
        # becomes image_urls, so a newly-added image is the next #imageN.
        images_override = _gallery_to_urls(frame_images_gallery, upload_video) or None

        updated_scene = _process_single_scene(target_scene_data, asset_urls, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality,
                                              prompt_override=frame_prompt_override, image_urls_override=images_override,
                                              motion_graphic=is_motion_graphic(production_type))
        
        for idx, old_scene in enumerate(current_storyboard_state):
            if str(old_scene["scene_number"]) == str(scene_num_to_regen):
                current_storyboard_state[idx] = updated_scene
                break
                
        gallery_format = _frames_to_gallery(current_storyboard_state)
        # Autosave the updated storyboard so the edit survives a restart.
        session_store.save_session(storyboard_frames=current_storyboard_state)
        gr.Info(f"Scene {scene_num_to_regen} successfully updated!")
        return gallery_format, current_storyboard_state
    except Exception as e:
        print(f"Single scene regeneration crash: {e}")
        return gr.update(), current_storyboard_state

def _payload_summary(args: dict, skip: tuple, header_lines: list) -> str:
    """Render everything in a sent payload except the fields already shown in
    their own editable boxes."""
    lines = list(header_lines)
    for k, v in (args or {}).items():
        if k in skip:
            continue
        text = str(v)
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"**{k}:** `{text}`")
    return "\n\n".join(lines)

def load_frame_inputs(storyboard_state, scene_number):
    """Show the EXACT inputs that produced a storyboard frame.

    Returns the editable prompt, a GALLERY of the input images each captioned
    #image1..#imageN (so the user sees the actual pictures, not URLs), a hint
    telling them what number a newly-uploaded image will take, and a settings
    summary.
    """
    empty = (gr.update(), gr.update(), gr.update(), gr.update())
    if not storyboard_state:
        gr.Warning("No storyboard yet — render the storyboard first.")
        return empty
    scene = next((s for s in storyboard_state
                  if str(s.get("scene_number")) == str(scene_number)), None)
    if not scene:
        gr.Warning(f"Scene {scene_number} not found.")
        return empty

    args = scene.get("image_args") or {}
    if not args:
        return ("", [],
                "_No stored inputs for this scene. Re-render the storyboard once so the inputs get recorded._",
                "")

    urls = args.get("image_urls", []) or []
    # Caption each existing input image with the tag the prompt uses to reference it.
    gallery = [(u, f"#image{i}") for i, u in enumerate(urls, start=1)]
    next_tag = f"#image{len(urls) + 1}"
    hint = (f"Above are the images sent for this scene. Any image you upload below "
            f"becomes **{next_tag}** — reference it in the prompt as `{next_tag}`.")

    header = [f"**Endpoint:** `{scene.get('image_endpoint', 'n/a')}`"]
    if scene.get("image_error"):
        header.append(f"**⚠ Last error:** {scene['image_error']}")
    qa = scene.get("qa_keyframe") or {}
    if qa.get("status") == "mismatch":
        header.append(f"**⚠ Keyframe QA:** prompt/frame mismatch — {qa.get('reason', '')}. "
                      f"Consider re-picking the keyframe time or editing the prompt below.")
    return (args.get("prompt", ""), gallery, hint,
            _payload_summary(args, ("prompt", "image_urls"), header))

def load_segment_inputs(segments_state, scene_number):
    """Show the EXACT payload that produced a video segment, loaded into the
    existing override boxes."""
    if not segments_state:
        gr.Warning("No rendered segments yet — run 'Render Final Commercial' first.")
        return gr.update(), gr.update(), gr.update()
    seg = next((s for s in segments_state
                if str(s.get("scene_number")) == str(scene_number)
                or str(s.get("idx", -1) + 1) == str(scene_number)), None)
    if not seg:
        gr.Warning(f"Segment {scene_number} not found.")
        return gr.update(), gr.update(), gr.update()

    args = seg.get("args") or {}
    header = [f"**Endpoint:** `{seg.get('endpoint', 'n/a')}`",
              f"**Status:** `{seg.get('status', 'n/a')}`"]
    if seg.get("error"):
        header.append(f"**⚠ Error:** {seg['error']}")
    img = (args.get("image_url", "") or args.get("start_image_url", "")
           or (args.get("image_urls") or [""])[0])
    # Make the frame roles explicit so start/end can't be mistaken for each other.
    end_img = args.get("end_image_url", "")
    header.append(f"**▶ START frame (first frame of this clip):** {img or '—'}")
    header.append(f"**⏹ END frame (locked last frame):** {end_img or '— free ending (no lock)'}")
    return (args.get("prompt", ""), img,
            _payload_summary(args, ("prompt", "image_url", "image_urls",
                                    "start_image_url", "end_image_url"), header))

def _frames_to_gallery(storyboard_frames):
    """Rebuild the storyboard gallery (image, caption) list from frame state,
    including the ⚠ keyframe-QA flag. Shared by generation and session restore."""
    gallery = []
    for frame in (storyboard_frames or []):
        if frame.get("generated_image_url"):
            display_text = frame.get('spoken_dialogue') or frame.get('sfx_music', '')
            qa = frame.get("qa_keyframe") or {}
            prefix = "⚠ " if qa.get("status") == "mismatch" else ""
            gallery.append((frame["generated_image_url"],
                            f"{prefix}Scene {frame['scene_number']}: {display_text}"))
    return gallery


def restore_session():
    """On app load, repopulate the storyboard/segment state + gallery from the last
    saved session so a server restart doesn't lose an expensive storyboard.
    Returns (gallery, storyboard_state, segments_state, status_markdown)."""
    data = session_store.load_session()
    frames = data.get("storyboard_frames") or []
    segments = data.get("video_segments") or []
    if not frames:
        return gr.update(), None, None, gr.update()
    age = session_store.session_age_str()
    note = (f"♻️ Restored previous session ({len(frames)} storyboard frames"
            + (f", saved {age}" if age else "") + "). Your work was not lost.")
    gr.Info(note)
    return _frames_to_gallery(frames), frames, (segments or None), note


def preview_timeline(storyboard_frames, target_duration):
    """Build the pre-generation timeline (free, no model calls) and lay it into the
    fixed pool of per-scene duration editors. Returns, in order:
      [status_markdown] + [row_visibility ...] + [row_label ...] + [row_value ...]
    """
    n_updates = lambda vis, lab, val: (vis + lab + val)
    hidden_rows = [gr.update(visible=False)] * MAX_TIMELINE_ROWS
    blank_labels = [gr.update()] * MAX_TIMELINE_ROWS
    blank_vals = [gr.update()] * MAX_TIMELINE_ROWS

    if not storyboard_frames:
        gr.Warning("Render the storyboard first, then preview the timeline.")
        return ["_No storyboard yet._", *n_updates(hidden_rows, blank_labels, blank_vals)]

    preview = build_timeline_preview(storyboard_frames, int(target_duration))
    rows = preview["rows"][:MAX_TIMELINE_ROWS]

    vis, labels, vals = [], [], []
    for i in range(MAX_TIMELINE_ROWS):
        if i < len(rows):
            r = rows[i]
            vis.append(gr.update(visible=True))
            labels.append(gr.update(value=f"**Scene {r['scene_number']}** · {r['dialogue_words']} words · "
                                          f"_{r['prompt_snippet'] or '(no prompt)'}_"))
            vals.append(gr.update(value=r["planned_duration"]))
        else:
            vis.append(gr.update(visible=False))
            labels.append(gr.update())
            vals.append(gr.update())

    status = _timeline_status([r["planned_duration"] for r in rows], preview["target"], len(rows))
    return [status, *n_updates(vis, labels, vals)]

def _timeline_status(durations, target, n_scenes):
    """Human-readable timeline summary + validity, shared by preview and live edits."""
    lo, hi = int(MODEL_MIN_CHUNK), int(MODEL_MAX_CHUNK)
    total = sum(clamp_chunk(d) for d in durations)
    bad = [i + 1 for i, d in enumerate(durations) if not (lo <= round(d) <= hi)]
    min_possible = n_scenes * lo
    target_txt = f"target {target}s, " if target else ""
    lines = [f"**Planned total: {total}s**  ({target_txt}{n_scenes} scenes)"]
    if bad:
        lines.append(f"⛔ Scenes {bad} are outside the model limit ({lo}–{hi}s each). "
                     f"Fix them to enable rendering.")
    if target and min_possible > target:
        lines.append(f"⚠ With {n_scenes} scenes the shortest possible ad is {min_possible}s "
                     f"(each chunk ≥{lo}s), so {target}s isn't reachable without fewer scenes.")
    if not bad:
        lines.append("✅ Timeline is valid — you can render.")
    return "\n\n".join(lines)

def _timeline_is_valid(durations):
    lo, hi = int(MODEL_MIN_CHUNK), int(MODEL_MAX_CHUNK)
    return all(lo <= round(d) <= hi for d in durations)

def _format_cost(tracker, segments=None):
    cost = tracker.total()
    lines = [f"Video gen: ${cost['video_generation_confirmed']}",
             f"Images/LLM: ${cost['images_and_llm_estimated']}",
             f"Total: ${cost['total_estimated']}"]
    if tracker.max_budget_usd > 0:
        lines.append(f"Budget cap: ${tracker.max_budget_usd:.2f}")
    if segments:
        from app.new_pipeline.cost_tracker import estimate_segment_cost
        lines.append("\nPer-scene (rendered):")
        for s in sorted(segments, key=lambda x: x.get("idx", 0)):
            if s.get("status") == "ok":
                c = estimate_segment_cost(s.get("model_id", ""), s.get("duration", 0))
                lines.append(f"  • Scene {s.get('idx', 0)+1}: {s.get('duration', 0)}s ~${c:.3f}")
            else:
                lines.append(f"  • Scene {s.get('idx', 0)+1}: failed — not billed")
    return "\n".join(lines)

def step4_generate_final(api_key, storyboard_frames, final_instructions, model_choice, duration, aspect_ratio, resolution, raw_script_text, production_type, max_budget, *timeline_vals):
    set_fal_key(api_key)
    if not storyboard_frames:
        return None, "Error: No storyboard frames found. Please complete Step 3.", None, gr.update()

    # Per-scene durations from the timeline editor pool (first N rows for N scenes).
    # If the user reviewed/edited them we honor them; if not (all blank), we fall
    # back to the automatic plan rather than blocking the render. Block ONLY when
    # the user actually supplied an out-of-range value.
    n = len(storyboard_frames)
    raw_vals = list(timeline_vals[:n])
    supplied = [v for v in raw_vals if v not in (None, "")]
    overrides = None
    if supplied:
        # Some durations were provided. Require ALL n to be present and valid so we
        # don't render a mix of edited + missing scenes.
        if len(supplied) != n:
            gr.Warning("Some scenes have no duration. Click 'Preview Timeline' to fill them all in.")
            return None, "⛔ Preview the timeline so every scene has a duration, then render.", None, gr.update()
        try:
            overrides = [float(v) for v in supplied]
        except (TypeError, ValueError):
            gr.Warning("Durations must be numbers.")
            return None, "⛔ One or more durations aren't valid numbers.", None, gr.update()
        if not _timeline_is_valid(overrides):
            lo, hi = int(MODEL_MIN_CHUNK), int(MODEL_MAX_CHUNK)
            gr.Warning(f"Every scene must be {lo}–{hi}s before rendering.")
            return None, (f"⛔ Some scene durations are outside the model limit ({lo}–{hi}s). "
                          f"Fix them in the timeline and try again."), None, gr.update()

    tracker = CostTracker(max_budget_usd=max_budget)
    tracker.add_analysis_call()
    # Price the images with the model/size/quality actually used in step 3
    # (stamped onto each frame). Falls back to defaults for older state.
    _first = storyboard_frames[0] if storyboard_frames else {}
    tracker.add_image_generation_calls(
        len(storyboard_frames),
        image_model=_first.get("image_model", DEFAULT_IMAGE_MODEL),
        image_size=_first.get("image_size", ""),
        image_quality=_first.get("image_quality", "high"),
    )
    # Bill the cheap keyframe-QA vision call for each scene that ran it (step 3).
    for _f in storyboard_frames:
        if _f.get("qa_keyframe"):
            tracker.add_caption_call()

    try:
        result = generate_final_video(
            storyboard_frames=storyboard_frames, final_instructions=final_instructions,
            model_id=model_choice, cost_tracker=tracker, total_duration=duration,
            aspect_ratio=aspect_ratio, resolution=resolution, raw_script_text=raw_script_text,
            production_type=production_type, duration_overrides=overrides
        )
        # result is a dict: partial video + per-segment state + status report.
        report = f"{result['status_report']}\n\n{_format_cost(tracker, result.get('segments'))}"
        # Pre-select the first failed scene in the re-render box, if any.
        failed = result.get("failed_scene_numbers") or []
        # gr.Number requires an int; scene_number may be a string -> round() crashes.
        seg_choice = gr.update()
        if failed:
            try:
                seg_choice = gr.update(value=int(failed[0]))
            except (TypeError, ValueError):
                seg_choice = gr.update()
        # Autosave storyboard + rendered segments so a restart keeps both.
        session_store.save_session(storyboard_frames=storyboard_frames,
                                   video_segments=result.get("segments"))
        return result["final_video_path"], report, result["segments"], seg_choice
    except BudgetExceeded as be:
        # Nothing was rendered — refused up-front. Images/analysis already billed.
        gr.Warning(str(be))
        return None, f"⛔ {be}\n\n{_format_cost(tracker)}", None, gr.update()
    except Exception as e:
        return None, f"Generation Failed: {str(e)}", None, gr.update()

def step4_add_music(current_video, music_file, duck_under_voice):
    """Post-step: lay a continuous music bed under the finished ad (ducked under
    voice). Fail-open — returns the existing video if no music / on any error."""
    if not current_video:
        gr.Warning("Render the final commercial first, then add music.")
        return current_video, "No rendered video to score yet."
    if not music_file:
        gr.Warning("Upload a music track to add a bed.")
        return current_video, "No music track supplied."
    from app.new_pipeline.audio_post import add_music_bed
    try:
        scored = add_music_bed(current_video, music_file, duck_under_voice=duck_under_voice)
        if scored == current_video:
            return current_video, "Music bed could not be applied — original audio kept."
        gr.Info("Music bed added.")
        return scored, ("Music bed added"
                        + (" (ducked under voice)." if duck_under_voice else "."))
    except Exception as e:
        return current_video, f"Music step failed: {e}"

def step4_rerender_segment(api_key, segments_state, scene_number, resolution, prompt_override, image_override):
    """Re-render ONE failed/edited video segment without touching the others."""
    set_fal_key(api_key)
    if not segments_state:
        return None, "No rendered segments found. Run 'Render Final Commercial' first.", segments_state
    tracker = CostTracker()
    try:
        result = regenerate_single_segment(
            segments=segments_state, scene_number=scene_number, cost_tracker=tracker,
            resolution=resolution, prompt_override=prompt_override, image_url_override=image_override
        )
        report = f"{result['status_report']}\n\n{_format_cost(tracker)}"
        session_store.save_session(video_segments=result.get("segments"))
        return result["final_video_path"], report, result["segments"]
    except Exception as e:
        return None, f"Re-render Failed: {str(e)}", segments_state

# Force dark mode for EVERY visitor. Gradio picks its light or dark variable
# set from the browser's prefers-color-scheme, but our custom CSS hardcodes
# dark backgrounds — on a light-mode browser that mix rendered black labels on
# near-black panels and white textboxes. Redirecting onto ?__theme=dark makes
# the app look identical on every machine.
FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

# --- UI LAYOUT ---
with gr.Blocks(title="AdForge Studio Pro", theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="header-container"):
        gr.Markdown("# ⚡ AdForge Studio Pro")
        gr.Markdown("Turn any reference ad into your own branded video — in four steps.")

        with gr.Row():
            api_key_input = gr.Textbox(
                label="🔑 FAL API Key",
                type="password",
                placeholder="Paste your fal_… key to authorize generation",
                elem_classes="panel-box"
            )
        # Shown when a previous session (storyboard) is auto-restored after a restart.
        restore_banner = gr.Markdown(value="", visible=True)

    state_storyboard_frames = gr.State()
    state_video_segments = gr.State()   # per-segment render state for chunk recovery

    with gr.Tabs():
        with gr.TabItem("1 · Strategy & Assets"):
            gr.Markdown("Fetch a reference ad, add your brand assets, and generate the editable script.")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 📥 Source Material")
                    ad_url = gr.Textbox(label="Reference Ad URL", placeholder="Paste an inspiration video link…", show_label=False)
                    btn_download = gr.Button("Fetch Source", variant="secondary")
                    gr.Markdown("_…or drag & drop / upload a local video below._")
                    # gr.Video accepts a direct upload too — either path fills this box.
                    preview_video = gr.Video(label="Source Video (paste a URL above, or upload here)", height=300)

                    gr.Markdown("### ⚙️ Production Settings")
                    # (label, value): values stay prefix-free so downstream checks
                    # match; ✅ = supported, 🚧 = not yet.
                    production_choices = [
                        ("✅ On-Screen Actor / Dialogue", "Narrative Ad (On-Screen Actor / Dialogue)"),
                        ("✅ Voiceover focused", "Narrative Ad (Voiceover focused)"),
                        ("🚧 No Voiceover — Text / Music", "Narrative Ad (No Voiceover - Text/Music focused)"),
                        ("🚧 Motion Graphic (SFX & Music)", "Motion Graphic (SFX & Music focused)"),
                    ]
                    production_type = gr.Dropdown(choices=production_choices, value="Narrative Ad (On-Screen Actor / Dialogue)", label="Format")
                    duration_choice = gr.Slider(minimum=3, maximum=60, step=1, value=15, label="Target Duration (sec)")

                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 🎨 Brand Assets & Directives")
                    uploaded_assets = gr.File(label="Brand Assets — faces, logos, UI screens", file_count="multiple", type="filepath", height=150)
                    asset_list_display = gr.Markdown(
                        value="_No assets uploaded yet. Upload files, then refer to them as **Asset 1**, **Asset 2**, … in your Creative Directive._"
                    )
                    creative_directive = gr.Textbox(label="Creative Directive", placeholder="E.g. 'Asset 1 is my UI dashboard, Asset 2 my logo. Swap the original product for Asset 1.'", lines=4)

                    btn_extract = gr.Button("🧠 Analyze Strategy & Compute Pacing", variant="primary", interactive=False)

            with gr.Row(elem_classes="panel-box"):
                extracted_prompts = gr.Textbox(label="AI Strategy & Script (editable)", lines=12, interactive=True,
                                               placeholder="Your generated storyboard script appears here after analysis. Edit it freely before rendering.")
                
        with gr.TabItem("2 · Storyboard"):
            gr.Markdown("Pick an image model and style, then render a still frame for each scene. "
                        "You can re-render any single scene without redoing the rest.")
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    # --- Image model selection ---
                    image_model_choice = gr.Dropdown(
                        choices=[(m["label"], key) for key, m in IMAGE_MODELS.items()],
                        value=DEFAULT_IMAGE_MODEL, label="Image Model")

                    # Aspect ratio is the OUTPUT/video aspect ratio (always used by
                    # the final video render, and by Nano Banana for images). Keep it
                    # always visible. GPT Image 2 additionally exposes image_size +
                    # quality, which only appear when GPT is selected.
                    aspect_ratio_choice = gr.Dropdown(
                        choices=["16:9", "9:16", "1:1"], value="9:16",
                        label="Aspect Ratio (Output / Video)")
                    _gpt = IMAGE_MODELS["gpt-image-2"]
                    image_size_choice = gr.Dropdown(
                        choices=_gpt["sizes"], value=_gpt["sizes"][0],
                        label="Image Size (GPT Image 2)", visible=False)
                    image_quality_choice = gr.Dropdown(
                        choices=_gpt["qualities"], value="medium",
                        label="Image Quality (GPT Image 2 — 'medium' costs less)", visible=False)

                    def _toggle_image_controls(model_key):
                        supports_size = IMAGE_MODELS.get(model_key, {}).get("supports_size", False)
                        return (
                            gr.update(visible=supports_size),  # image_size
                            gr.update(visible=supports_size),  # image_quality
                        )
                    image_model_choice.change(
                        fn=_toggle_image_controls, inputs=[image_model_choice],
                        outputs=[image_size_choice, image_quality_choice])

                    font_choices = [
                        ("Keep Existing Text (use the reference's own fonts)", "KEEP_EXISTING_TEXT"),
                        ("No Text (remove all text, including the original)", "NO_TEXT"),
                        ("Chunky Display (Bold & Playful)", "chunky_display"),
                        ("Elegant Script Mix (High Fashion)", "elegant_script_mix"),
                        ("Streetwear Graffiti (Grungy Urban)", "streetwear_graffiti"),
                        ("Brutalist Geometric (Sharp & Minimal)", "brutalist_geometric"),
                        ("Retro 3D Editorial (Vintage 70s)", "retro_3d_editorial"),
                        ("Psychedelic Bubble (Curvy & Fun)", "psychedelic_bubble"),
                        ("Liquid Cursive (Fluid Abstract)", "liquid_cursive"),
                        ("Raw Expressionist (Jagged Papercut)", "raw_expressionist")
                    ]
                    font_style_choice = gr.Dropdown(
                        choices=font_choices, value="chunky_display", label="Typography Overlay Style",
                        info="Motion graphics: pick 'Keep Existing Text' so the reference's own "
                             "fonts survive and no extra type is stacked on top. 'No Text' erases "
                             "the original lettering too.")

                    qa_toggle = gr.Checkbox(
                        value=True, label="🔎 Keyframe QA (flag prompt/frame mismatches)",
                        info="Cheap vision check per scene. Flags only — never changes the render.")

                    btn_generate_storyboard = gr.Button("🎨 Render Storyboard", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("### 🔁 Re-render one scene")
                    scene_num_input = gr.Number(label="Scene number", precision=0, value=1)
                    btn_load_frame_inputs = gr.Button("🔍 Load inputs for this scene", variant="secondary")
                    frame_info = gr.Markdown(value="_Load a scene to see exactly what was sent to the image model._")
                    frame_prompt_edit = gr.Textbox(label="Prompt sent (edit to change it)", lines=6,
                                                   placeholder="Click 'Load inputs' to see the actual prompt used.")
                    # Input images as labelled thumbnails (#image1, #image2, …).
                    # interactive=True adds the built-in "+" upload tile inside the
                    # gallery — a dropped image becomes the next #imageN.
                    frame_images_gallery = gr.Gallery(label="Input images (use + to add the next #image)",
                                                      columns=3, height=160, object_fit="contain",
                                                      interactive=True)
                    frame_extra_hint = gr.Markdown(value="")
                    btn_regen_single = gr.Button("Re-render this scene", variant="secondary")

                with gr.Column(scale=3):
                    storyboard_gallery = gr.Gallery(label="Storyboard", columns=3, height=600)
                    
        with gr.TabItem("3 · Final Render"):
            gr.Markdown("Animate the storyboard into the final video. If a scene fails, fix and re-render "
                        "just that one below — the others are kept.")
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(choices=list(MODELS.keys()), value="gemini-omni-flash", label="Animation Engine")
                    gr.Markdown(
                        "💡 **Motion graphics?** Pick an **end-frame-lock** engine "
                        "(*Kling O3* — cheapest, or *Wan 2.7*). They force each clip to "
                        "end exactly on the next scene's frame, so clips join with no cut.",
                        elem_classes="hint")
                    resolution_choice = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    final_instructions = gr.Textbox(label="Global Override Instructions", placeholder="E.g. Keep lighting cinematic, end with logo fade.", lines=3)
                    max_budget_input = gr.Number(
                        label="💵 Max budget per run (USD, 0 = no cap)", value=2.0, minimum=0, step=0.5,
                        info="Refuses to start the video render if the projected cost exceeds this.")

                    gr.Markdown("---")
                    gr.Markdown("### ⏱️ Preview & adjust timeline")
                    gr.Markdown(
                        "See how long the ad will be BEFORE spending. Each scene shows "
                        "its planned duration — edit any of them, then render. Chunks "
                        f"must be {int(MODEL_MIN_CHUNK)}–{int(MODEL_MAX_CHUNK)}s (model limit)."
                    )
                    btn_preview_timeline = gr.Button("📋 Preview Timeline", variant="secondary")
                    timeline_status = gr.Markdown(value="_Click 'Preview Timeline' to plan durations._")
                    # Fixed pool of per-scene duration editors; shown up to scene count.
                    # Default value is None (blank) so "not previewed yet" is detectable
                    # and never sends a stale 5 into the render.
                    timeline_labels, timeline_numbers, timeline_rows = [], [], []
                    for _i in range(MAX_TIMELINE_ROWS):
                        with gr.Row(visible=False) as _row:
                            _lbl = gr.Markdown(value="", elem_classes="timeline-label")
                            _num = gr.Number(label="sec", value=None, precision=0,
                                             minimum=int(MODEL_MIN_CHUNK), maximum=int(MODEL_MAX_CHUNK),
                                             scale=0)
                        timeline_rows.append(_row)
                        timeline_labels.append(_lbl)
                        timeline_numbers.append(_num)

                    btn_generate_final = gr.Button("🎬 Render Final Commercial", variant="primary")
                    cost_output = gr.Textbox(label="Render Status & Cost", lines=6, interactive=False)

                    gr.Markdown("---")
                    gr.Markdown("### 🩹 Segment Recovery")
                    gr.Markdown(
                        "If a scene failed, its reason appears above. Fix the cause "
                        "below (edit the prompt and/or paste a new base image URL), "
                        "then re-render just that segment — other scenes are kept."
                    )
                    rerender_scene_num = gr.Number(label="Scene # to re-render", precision=0, value=1)
                    btn_load_segment_inputs = gr.Button("🔍 Load inputs for this segment", variant="secondary")
                    segment_info = gr.Markdown(value="_Load a segment to see exactly what was sent to the video model._")
                    rerender_prompt = gr.Textbox(label="Corrected Prompt (optional)", lines=3,
                                                 placeholder="Leave blank to reuse the original prompt.")
                    rerender_image_url = gr.Textbox(label="Replacement Base Image URL (optional)",
                                                    placeholder="Paste a new keyframe/asset URL if the image was the problem.")
                    btn_rerender_segment = gr.Button("🔁 Re-Render This Segment", variant="secondary")

                    gr.Markdown("---")
                    gr.Markdown("### 🎵 Music Bed")
                    gr.Markdown(
                        "Lay one continuous track under the finished ad. With ducking "
                        "on, the music automatically dips under any dialogue and swells "
                        "back in the gaps. Runs on the rendered video — no re-render."
                    )
                    music_upload = gr.Audio(label="Music track", type="filepath")
                    duck_toggle = gr.Checkbox(value=True, label="Duck music under voice")
                    btn_add_music = gr.Button("🎚️ Add Music Bed", variant="secondary")

                with gr.Column(scale=3):
                    final_video = gr.Video(label="Final AdForge Export", height=600)

    # WIRING
    btn_download.click(
        fn=step1_download,
        inputs=[ad_url],
        outputs=[preview_video, btn_extract]
    )

    # Loading a local video directly (upload or drag-drop) also readies the flow.
    preview_video.upload(fn=on_source_ready, inputs=[preview_video], outputs=[btn_extract])
    preview_video.change(fn=on_source_ready, inputs=[preview_video], outputs=[btn_extract])
    
    # Keep the numbered asset list in sync so the user always knows which upload
    # is "Asset 1", "Asset 2", … before writing the Creative Directive.
    uploaded_assets.change(
        fn=render_asset_list,
        inputs=[uploaded_assets],
        outputs=[asset_list_display]
    )

    btn_extract.click(
        fn=step2_extract_scenes,
        inputs=[api_key_input, preview_video, creative_directive, duration_choice, production_type, uploaded_assets],
        outputs=[extracted_prompts]
    )
    
    btn_generate_storyboard.click(
        fn=step3_generate_storyboard,
        inputs=[api_key_input, extracted_prompts, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice, image_model_choice, image_size_choice, image_quality_choice, production_type, qa_toggle],
        outputs=[storyboard_gallery, state_storyboard_frames]
    )

    # Inspect: load the prompt + labelled input-image thumbnails for this scene.
    btn_load_frame_inputs.click(
        fn=load_frame_inputs,
        inputs=[state_storyboard_frames, scene_num_input],
        outputs=[frame_prompt_edit, frame_images_gallery, frame_extra_hint, frame_info]
    )

    btn_regen_single.click(
        fn=step3_regenerate_single,
        inputs=[api_key_input, extracted_prompts, scene_num_input, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice, state_storyboard_frames, image_model_choice, image_size_choice, image_quality_choice, frame_prompt_edit, frame_images_gallery, production_type],
        outputs=[storyboard_gallery, state_storyboard_frames]
    )
    
    # Pre-generation timeline: plan durations (free) and lay them into the editors.
    btn_preview_timeline.click(
        fn=preview_timeline,
        inputs=[state_storyboard_frames, duration_choice],
        outputs=[timeline_status, *timeline_rows, *timeline_labels, *timeline_numbers]
    )

    # Live-recompute the status/validity whenever any duration is edited.
    def _recalc_timeline(storyboard_frames, *vals):
        if not storyboard_frames:
            return gr.update()
        n = len(storyboard_frames)
        durs = [v for v in vals[:n] if v not in (None, "")]
        if len(durs) != n:
            return gr.update()
        return _timeline_status(durs, None, n)
    for _num in timeline_numbers:
        _num.change(
            fn=_recalc_timeline,
            inputs=[state_storyboard_frames, *timeline_numbers],
            outputs=[timeline_status]
        )

    btn_generate_final.click(
        fn=step4_generate_final,
        inputs=[api_key_input, state_storyboard_frames, final_instructions, model_choice, duration_choice, aspect_ratio_choice, resolution_choice, extracted_prompts, production_type, max_budget_input, *timeline_numbers],
        outputs=[final_video, cost_output, state_video_segments, rerender_scene_num]
    )

    # Inspect: load the exact payload that produced this segment into the boxes above.
    btn_load_segment_inputs.click(
        fn=load_segment_inputs,
        inputs=[state_video_segments, rerender_scene_num],
        outputs=[rerender_prompt, rerender_image_url, segment_info]
    )

    btn_rerender_segment.click(
        fn=step4_rerender_segment,
        inputs=[api_key_input, state_video_segments, rerender_scene_num, resolution_choice, rerender_prompt, rerender_image_url],
        outputs=[final_video, cost_output, state_video_segments]
    )

    btn_add_music.click(
        fn=step4_add_music,
        inputs=[final_video, music_upload, duck_toggle],
        outputs=[final_video, cost_output]
    )

    # On page load, restore the last saved session (storyboard + segments) so a
    # server restart / code change never loses an expensive storyboard.
    demo.load(
        fn=restore_session,
        inputs=None,
        outputs=[storyboard_gallery, state_storyboard_frames, state_video_segments, restore_banner]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, js=FORCE_DARK_JS)