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

from app.new_pipeline.analyzer import analyze_and_extract_scenes
from app.new_pipeline.frame_designer import generate_storyboard_frames, _process_single_scene, parse_extracted_scenes, IMAGE_MODELS, DEFAULT_IMAGE_MODEL
from app.new_pipeline.generator import generate_final_video, regenerate_single_segment

# --- THEME & CSS ---
# Gradio 6 renamed many internal button classes; target buttons by variant via
# the stable `button` element + Gradio's own `.primary`/`.secondary` variant
# classes rather than the old `.gr-button-*` names (which silently no-op now).
CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; max-width: 1280px; margin: 0 auto; }

#header-container {
  text-align: center; margin-bottom: 24px; padding: 28px 20px;
  background: linear-gradient(120deg, #1e1b4b 0%, #312e81 55%, #4c1d95 100%);
  border-radius: 16px; border: 1px solid #4338ca;
  box-shadow: 0 8px 28px rgba(49, 46, 129, 0.35);
}
#header-container h1 { font-weight: 800; letter-spacing: -0.02em; }
h1, h2, h3, h4, p, label span { color: #f3f4f6 !important; }

.panel-box {
  border-radius: 14px !important; border: 1px solid #1f2937 !important;
  background-color: #111827 !important; padding: 18px !important;
  box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}

/* Buttons — Gradio 6 uses variant classes on the <button> element */
button.primary, .primary button {
  background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
  border: none !important; font-weight: 600 !important;
  box-shadow: 0 2px 12px rgba(139, 92, 246, 0.45) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
button.primary:hover, .primary button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 18px rgba(139, 92, 246, 0.7) !important;
}
button.secondary, .secondary button {
  background: #1f2937 !important; border: 1px solid #374151 !important;
  transition: background 0.15s ease;
}
button.secondary:hover, .secondary button:hover { background: #374151 !important; }

/* Tabs */
.tab-nav button { font-weight: 600 !important; }
.tab-nav button.selected { color: #a78bfa !important; border-bottom: 2px solid #8b5cf6 !important; }

/* Step helper text under tab titles */
.step-hint p { color: #9ca3af !important; font-size: 0.9em; margin: 2px 0 10px 0; }
.status-ok textarea { color: #86efac !important; }
"""

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate").set(
    body_background_fill="#0b0f19", block_background_fill="#111827",
    block_border_color="#1f2937", button_primary_background_fill="#6366f1",
    block_radius="14px", button_large_radius="10px",
)

def set_fal_key(api_key: str):
    if not api_key or not api_key.strip():
        raise gr.Error("Missing API key — paste your FAL key in the header to authorize generation.")
    os.environ["FAL_KEY"] = api_key.strip()

def validate_key(api_key: str):
    """Lightweight key presence check with a visible confirmation."""
    if api_key and api_key.strip():
        os.environ["FAL_KEY"] = api_key.strip()
        return gr.update(value="🔑 Key set", visible=True)
    return gr.update(value="", visible=False)

def step1_download(url, progress=gr.Progress()):
    if not url or not url.strip():
        raise gr.Error("Please paste a reference ad URL first.")
    progress(0.2, desc="Fetching source video…")
    try:
        local_path = download_video(url.strip())
    except Exception as e:
        raise gr.Error(f"Could not fetch that video: {e}")
    progress(1.0, desc="Done")
    gr.Info("Source video fetched. You can now analyze the strategy.")
    return local_path, gr.update(interactive=True)

def step2_extract_scenes(api_key, local_video_path, creative_directive, target_duration, production_type, progress=gr.Progress()):
    set_fal_key(api_key)
    if not local_video_path:
        raise gr.Error("Please fetch a source video first (Step 1).")
    try:
        progress(0.2, desc="Uploading video to fal…")
        hosted_video_url = upload_video(local_video_path)
        progress(0.5, desc="Analyzing strategy & computing pacing…")
        extracted_text = analyze_and_extract_scenes(
            video_url=hosted_video_url,
            target_duration=int(target_duration),
            creative_directive=creative_directive,
            production_type=production_type,
            # The creative directive IS the asset instruction set (it describes what
            # each uploaded asset is and how to swap it in). Feed it to both prompts
            # so the VLM actually populates assigned_assets instead of leaving [] .
            asset_instructions=creative_directive
        )
        progress(1.0, desc="Done")
        gr.Info("Strategy generated. Review/edit the script, then open the Storyboard tab.")
        return extracted_text
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Analysis failed: {e}")

def step3_generate_storyboard(api_key, edited_prompts, uploaded_assets, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality, progress=gr.Progress()):
    set_fal_key(api_key)
    if not edited_prompts or not edited_prompts.strip():
        raise gr.Error("No script found. Complete Step 1 (Strategy & Assets) first.")
    if not local_video_path:
        raise gr.Error("Source video missing — fetch it in Step 1 before rendering the storyboard.")
    asset_paths = uploaded_assets if uploaded_assets else []
    try:
        progress(0.15, desc="Extracting keyframes & compositing scenes…")
        storyboard_data = generate_storyboard_frames(edited_prompts, asset_paths, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality)

        # Stamp the image-gen settings onto each frame so the final cost estimate
        # (step 4) can price the images with the actual model/size/quality used.
        for frame in storyboard_data:
            frame["image_model"] = image_model
            frame["image_size"] = image_size
            frame["image_quality"] = image_quality

        # Build gallery captions (split dialogue/sfx schema).
        gallery_format = []
        for frame in storyboard_data:
            if frame.get("generated_image_url"):
                display_text = frame.get('spoken_dialogue') or frame.get('sfx_music', '')
                gallery_format.append((frame["generated_image_url"], f"Scene {frame['scene_number']}: {display_text}"))

        progress(1.0, desc="Done")
        rendered = len(gallery_format)
        total = len(storyboard_data)
        if rendered == 0:
            raise gr.Error("No scenes rendered. Check that the script parsed correctly and your key is valid.")
        if rendered < total:
            gr.Warning(f"Rendered {rendered}/{total} scenes — some failed. Use 'Re-render one scene' to retry them.")
        else:
            gr.Info(f"Storyboard ready — {rendered} scenes. Open the Final Render tab to animate.")
        return gallery_format, storyboard_data
    except gr.Error:
        raise
    except Exception as e:
        print(f"Storyboard compilation error: {str(e)}")
        raise gr.Error(f"Storyboard generation failed: {e}")

def step3_regenerate_single(api_key, edited_prompts, scene_num_to_regen, uploaded_assets, local_video_path, aspect_ratio, font_style, current_storyboard_state, image_model, image_size, image_quality):
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
                
        updated_scene = _process_single_scene(target_scene_data, asset_urls, local_video_path, aspect_ratio, font_style, image_model, image_size, image_quality)
        
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
                
        gr.Info(f"Scene {scene_num_to_regen} updated.")
        return gallery_format, current_storyboard_state
    except gr.Error:
        raise
    except Exception as e:
        print(f"Single scene regeneration crash: {e}")
        raise gr.Error(f"Could not re-render scene {scene_num_to_regen}: {e}")

def _format_cost(tracker):
    cost = tracker.total()
    return (
        f"| Item | Cost |\n"
        f"|---|---|\n"
        f"| Video generation | ${cost['video_generation_confirmed']} |\n"
        f"| Images + LLM (est.) | ${cost['images_and_llm_estimated']} |\n"
        f"| **Total (est.)** | **${cost['total_estimated']}** |"
    )

def step4_generate_final(api_key, storyboard_frames, final_instructions, model_choice, duration, aspect_ratio, resolution, raw_script_text, production_type, progress=gr.Progress()):
    set_fal_key(api_key)
    if not storyboard_frames:
        raise gr.Error("No storyboard frames found — complete the Storyboard step first.")
    progress(0.1, desc="Planning segment durations…")

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
        progress(0.3, desc="Rendering & stitching video scenes… (this can take a few minutes)")
        result = generate_final_video(
            storyboard_frames=storyboard_frames, final_instructions=final_instructions,
            model_id=model_choice, cost_tracker=tracker, total_duration=duration,
            aspect_ratio=aspect_ratio, resolution=resolution, raw_script_text=raw_script_text,
            production_type=production_type
        )
        progress(1.0, desc="Done")
        # result is a dict: partial video + per-segment state + status report.
        report = f"{result['status_report']}\n\n{_format_cost(tracker)}"
        failed = result.get("failed_scene_numbers") or []
        if failed:
            gr.Warning(f"{len(failed)} scene(s) failed. See status below and use Segment Recovery to fix them.")
        else:
            gr.Info("Final commercial rendered.")
        # Pre-select the first failed scene in the re-render box, if any.
        seg_choice = gr.update(value=failed[0]) if failed else gr.update()
        return result["final_video_path"], report, result["segments"], seg_choice
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Video generation failed: {e}")

def step4_rerender_segment(api_key, segments_state, scene_number, resolution, prompt_override, image_override, progress=gr.Progress()):
    """Re-render ONE failed/edited video segment without touching the others."""
    set_fal_key(api_key)
    if not segments_state:
        raise gr.Error("No rendered segments found — run 'Render Final Commercial' first.")
    tracker = CostTracker()
    try:
        progress(0.3, desc=f"Re-rendering scene {scene_number} & re-stitching…")
        result = regenerate_single_segment(
            segments=segments_state, scene_number=scene_number, cost_tracker=tracker,
            resolution=resolution, prompt_override=prompt_override, image_url_override=image_override
        )
        progress(1.0, desc="Done")
        report = f"{result['status_report']}\n\n{_format_cost(tracker)}"
        gr.Info(f"Scene {scene_number} re-rendered.")
        return result["final_video_path"], report, result["segments"]
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Re-render failed: {e}")

# --- UI LAYOUT ---
with gr.Blocks(title="AdForge Studio Pro", theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="header-container"):
        gr.Markdown("# ⚡ AdForge Studio Pro")
        gr.Markdown("Turn any reference ad into your own — AI concept transfer in four steps.")

        with gr.Row():
            with gr.Column(scale=5):
                api_key_input = gr.Textbox(
                    label="🔑 FAL API Key",
                    type="password",
                    placeholder="Paste your fal_… key to authorize generation",
                )
            with gr.Column(scale=1, min_width=110):
                key_status = gr.Textbox(value="", show_label=False, interactive=False,
                                        visible=False, container=False)

    state_storyboard_frames = gr.State()
    state_video_segments = gr.State()   # per-segment render state for chunk recovery

    with gr.Tabs():
        with gr.TabItem("1 · Strategy & Assets"):
            gr.Markdown("Fetch a reference ad, add your brand assets, and generate the editable script.",
                        elem_classes="step-hint")
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 📥 Source Material")
                    ad_url = gr.Textbox(label="Reference Ad URL", placeholder="Paste an inspiration video link…")
                    btn_download = gr.Button("Fetch Source", variant="secondary")
                    preview_video = gr.Video(label="Source Video", height=300)

                    gr.Markdown("### ⚙️ Production Settings")
                    # (label shown to user, value passed to pipeline). Values are
                    # prefix-free so they match production_type checks downstream;
                    # unsupported formats are marked but selectable for preview.
                    production_choices = [
                        ("✅ Narrative Ad (On-Screen Actor / Dialogue)", "Narrative Ad (On-Screen Actor / Dialogue)"),
                        ("✅ Narrative Ad (Voiceover focused)", "Narrative Ad (Voiceover focused)"),
                        ("🚧 Narrative Ad (No Voiceover — Text/Music)", "Narrative Ad (No Voiceover - Text/Music focused)"),
                        ("🚧 Motion Graphic (SFX & Music)", "Motion Graphic (SFX & Music focused)"),
                    ]
                    production_type = gr.Dropdown(choices=production_choices,
                                                  value="Narrative Ad (On-Screen Actor / Dialogue)",
                                                  label="Format Variant")
                    duration_choice = gr.Slider(minimum=3, maximum=60, step=1, value=15, label="Target Duration (sec)")

                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 🎨 Brand Assets & Directives")
                    uploaded_assets = gr.File(label="Assets (faces, logos, UI screens — drop or browse)", file_count="multiple", type="filepath", height=150)
                    creative_directive = gr.Textbox(label="Creative Directive", placeholder="E.g., 'Asset 1 is my UI dashboard. Asset 2 is my logo. Swap out the original product for Asset 1.'", lines=4)

                    btn_extract = gr.Button("🧠 Analyze Strategy & Compute Pacing", variant="primary", interactive=False)

            with gr.Row(elem_classes="panel-box"):
                extracted_prompts = gr.Textbox(label="AI Strategy & Dynamic Script (Editable)", lines=12, interactive=True)
                
        with gr.TabItem("2 · Storyboard"):
            gr.Markdown("Pick an image model and style, then render the still frames for each scene. "
                        "Re-render any single scene without redoing the rest.", elem_classes="step-hint")
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
                    gr.Markdown("Not happy with a single frame? Enter its number and re-render just that one.",
                                elem_classes="step-hint")
                    scene_num_input = gr.Number(label="Scene number", precision=0, value=1)
                    btn_regen_single = gr.Button("Re-render this scene", variant="secondary")

                with gr.Column(scale=3):
                    storyboard_gallery = gr.Gallery(label="Storyboard", columns=3, height=600)

        with gr.TabItem("3 · Final Render"):
            gr.Markdown("Animate the storyboard into the final video. If any scene fails, fix and re-render "
                        "just that one in Segment Recovery — the others are kept.", elem_classes="step-hint")
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(choices=list(MODELS.keys()), value="gemini-omni-flash", label="Animation Engine")
                    resolution_choice = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    final_instructions = gr.Textbox(label="Global Override Instructions", placeholder="e.g. Keep lighting cinematic, end with logo fade.", lines=3)
                    btn_generate_final = gr.Button("🎬 Render Final Commercial", variant="primary")
                    cost_output = gr.Markdown(value="", label="Render status & cost")

                    gr.Markdown("---")
                    gr.Markdown("### 🩹 Segment Recovery")
                    gr.Markdown(
                        "If a scene failed, its reason appears above. Fix the cause "
                        "(edit the prompt and/or paste a new base image URL), then "
                        "re-render just that segment — other scenes are kept.",
                        elem_classes="step-hint"
                    )
                    rerender_scene_num = gr.Number(label="Scene # to re-render", precision=0, value=1)
                    rerender_prompt = gr.Textbox(label="Corrected Prompt (optional)", lines=3,
                                                 placeholder="Leave blank to reuse the original prompt.")
                    rerender_image_url = gr.Textbox(label="Replacement Base Image URL (optional)",
                                                    placeholder="Paste a new keyframe/asset URL if the image was the problem.")
                    btn_rerender_segment = gr.Button("🔁 Re-Render This Segment", variant="secondary")

                with gr.Column(scale=3):
                    final_video = gr.Video(label="Final AdForge Export", height=600)

    # WIRING
    api_key_input.change(fn=validate_key, inputs=[api_key_input], outputs=[key_status])

    btn_download.click(
        fn=step1_download,
        inputs=[ad_url],
        outputs=[preview_video, btn_extract]
    )
    
    btn_extract.click(
        fn=step2_extract_scenes, 
        inputs=[api_key_input, preview_video, creative_directive, duration_choice, production_type], 
        outputs=[extracted_prompts]
    )
    
    btn_generate_storyboard.click(
        fn=step3_generate_storyboard,
        inputs=[api_key_input, extracted_prompts, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice, image_model_choice, image_size_choice, image_quality_choice],
        outputs=[storyboard_gallery, state_storyboard_frames]
    )

    btn_regen_single.click(
        fn=step3_regenerate_single,
        inputs=[api_key_input, extracted_prompts, scene_num_input, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice, state_storyboard_frames, image_model_choice, image_size_choice, image_quality_choice],
        outputs=[storyboard_gallery, state_storyboard_frames]
    )
    
    btn_generate_final.click(
        fn=step4_generate_final,
        inputs=[api_key_input, state_storyboard_frames, final_instructions, model_choice, duration_choice, aspect_ratio_choice, resolution_choice, extracted_prompts, production_type],
        outputs=[final_video, cost_output, state_video_segments, rerender_scene_num]
    )

    btn_rerender_segment.click(
        fn=step4_rerender_segment,
        inputs=[api_key_input, state_video_segments, rerender_scene_num, resolution_choice, rerender_prompt, rerender_image_url],
        outputs=[final_video, cost_output, state_video_segments]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)