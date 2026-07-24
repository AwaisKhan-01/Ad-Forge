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
from app.new_pipeline.cost_tracker import CostTracker

from app.new_pipeline.analyzer import analyze_and_extract_scenes, is_motion_graphic
from app.new_pipeline.frame_designer import generate_storyboard_frames, _process_single_scene, parse_extracted_scenes, IMAGE_MODELS, DEFAULT_IMAGE_MODEL
from app.new_pipeline.generator import generate_final_video, regenerate_single_segment

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

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="gray").set(
    body_background_fill="#0b0f19", block_background_fill="#111827",
    block_border_color="#1f2937", button_primary_background_fill="#6366f1"
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

def step3_generate_storyboard(api_key, edited_prompts, uploaded_assets, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality, production_type=""):
    set_fal_key(api_key)
    if not edited_prompts:
        return [], None
    asset_paths = uploaded_assets if uploaded_assets else []
    try:
        mg = is_motion_graphic(production_type)
        storyboard_data = generate_storyboard_frames(edited_prompts, asset_paths, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality, motion_graphic=mg)

        # Stamp the image-gen settings onto each frame so the final cost estimate
        # (step 4) can price the images with the actual model/size/quality used.
        for frame in storyboard_data:
            frame["image_model"] = image_model
            frame["image_size"] = image_size
            frame["image_quality"] = image_quality

        # --- SURGICAL FIX: Update Gallery Captions for Split Schema ---
        gallery_format = []
        for frame in storyboard_data:
            if frame.get("generated_image_url"):
                display_text = frame.get('spoken_dialogue') or frame.get('sfx_music', '')
                gallery_format.append((frame["generated_image_url"], f"Scene {frame['scene_number']}: {display_text}"))
                
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
                
        # --- SURGICAL FIX: Update Gallery Captions for Split Schema ---
        gallery_format = []
        for frame in current_storyboard_state:
            if frame.get("generated_image_url"):
                display_text = frame.get('spoken_dialogue') or frame.get('sfx_music', '')
                gallery_format.append((frame["generated_image_url"], f"Scene {frame['scene_number']}: {display_text}"))
                
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
    img = args.get("image_url", "") or (args.get("image_urls") or [""])[0]
    return (args.get("prompt", ""), img,
            _payload_summary(args, ("prompt", "image_url", "image_urls"), header))

def _format_cost(tracker):
    cost = tracker.total()
    return (f"Video gen: ${cost['video_generation_confirmed']}\n"
            f"Images/LLM: ${cost['images_and_llm_estimated']}\n"
            f"Total: ${cost['total_estimated']}")

def step4_generate_final(api_key, storyboard_frames, final_instructions, model_choice, duration, aspect_ratio, resolution, raw_script_text, production_type):
    set_fal_key(api_key)
    if not storyboard_frames:
        return None, "Error: No storyboard frames found. Please complete Step 3.", None, gr.update()

    tracker = CostTracker()
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

    try:
        result = generate_final_video(
            storyboard_frames=storyboard_frames, final_instructions=final_instructions,
            model_id=model_choice, cost_tracker=tracker, total_duration=duration,
            aspect_ratio=aspect_ratio, resolution=resolution, raw_script_text=raw_script_text,
            production_type=production_type
        )
        # result is a dict: partial video + per-segment state + status report.
        report = f"{result['status_report']}\n\n{_format_cost(tracker)}"
        # Pre-select the first failed scene in the re-render box, if any.
        failed = result.get("failed_scene_numbers") or []
        # gr.Number requires an int; scene_number may be a string -> round() crashes.
        seg_choice = gr.update()
        if failed:
            try:
                seg_choice = gr.update(value=int(failed[0]))
            except (TypeError, ValueError):
                seg_choice = gr.update()
        return result["final_video_path"], report, result["segments"], seg_choice
    except Exception as e:
        return None, f"Generation Failed: {str(e)}", None, gr.update()

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
        return result["final_video_path"], report, result["segments"]
    except Exception as e:
        return None, f"Re-render Failed: {str(e)}", segments_state

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
                        choices=["16:9", "9:16", "1:1"], value="16:9",
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
                        ("No Text", "NO_TEXT"),
                        ("Chunky Display (Bold & Playful)", "chunky_display"),
                        ("Elegant Script Mix (High Fashion)", "elegant_script_mix"),
                        ("Streetwear Graffiti (Grungy Urban)", "streetwear_graffiti"),
                        ("Brutalist Geometric (Sharp & Minimal)", "brutalist_geometric"),
                        ("Retro 3D Editorial (Vintage 70s)", "retro_3d_editorial"),
                        ("Psychedelic Bubble (Curvy & Fun)", "psychedelic_bubble"),
                        ("Liquid Cursive (Fluid Abstract)", "liquid_cursive"),
                        ("Raw Expressionist (Jagged Papercut)", "raw_expressionist")
                    ]
                    font_style_choice = gr.Dropdown(choices=font_choices, value="chunky_display", label="Typography Overlay Style")
                    
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
                    resolution_choice = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    final_instructions = gr.Textbox(label="Global Override Instructions", placeholder="E.g. Keep lighting cinematic, end with logo fade.", lines=3)
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
        inputs=[api_key_input, extracted_prompts, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice, image_model_choice, image_size_choice, image_quality_choice, production_type],
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
    
    btn_generate_final.click(
        fn=step4_generate_final,
        inputs=[api_key_input, state_storyboard_frames, final_instructions, model_choice, duration_choice, aspect_ratio_choice, resolution_choice, extracted_prompts, production_type],
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)