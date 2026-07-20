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
from app.new_pipeline.frame_designer import generate_storyboard_frames, _process_single_scene, parse_extracted_scenes
from app.new_pipeline.generator import generate_final_video

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

def step1_download(url):
    local_path = download_video(url)
    return local_path, gr.update(interactive=True)

def step2_extract_scenes(api_key, local_video_path, creative_directive, target_duration, production_type):
    set_fal_key(api_key)
    if not local_video_path:
        return "Error: Please download a video first."
    try:
        hosted_video_url = upload_video(local_video_path)
        extracted_text = analyze_and_extract_scenes(
            video_url=hosted_video_url, 
            target_duration=int(target_duration),
            creative_directive=creative_directive,
            production_type=production_type
        )
        return extracted_text
    except Exception as e:
        return f"PIPELINE ERROR: {str(e)}"

def step3_generate_storyboard(api_key, edited_prompts, uploaded_assets, local_video_path, aspect_ratio, font_style):
    set_fal_key(api_key)
    if not edited_prompts:
        return [], None
    asset_paths = uploaded_assets if uploaded_assets else []
    try:
        storyboard_data = generate_storyboard_frames(edited_prompts, asset_paths, local_video_path, aspect_ratio, font_style)
        
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

def step3_regenerate_single(api_key, edited_prompts, scene_num_to_regen, uploaded_assets, local_video_path, aspect_ratio, font_style, current_storyboard_state):
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
                
        updated_scene = _process_single_scene(target_scene_data, asset_urls, local_video_path, aspect_ratio, font_style)
        
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

def step4_generate_final(api_key, storyboard_frames, final_instructions, model_choice, duration, aspect_ratio, resolution, raw_script_text, production_type):
    set_fal_key(api_key)
    if not storyboard_frames:
        return None, "Error: No storyboard frames found. Please complete Step 3."
    
    tracker = CostTracker()
    tracker.add_analysis_call() 
    tracker.add_image_generation_calls(len(storyboard_frames))
    
    try:
        final_video_url = generate_final_video(
            storyboard_frames=storyboard_frames, final_instructions=final_instructions, 
            model_id=model_choice, cost_tracker=tracker, total_duration=duration, 
            aspect_ratio=aspect_ratio, resolution=resolution, raw_script_text=raw_script_text,
            production_type=production_type
        )
        cost = tracker.total()
        cost_text = f"Video gen: ${cost['video_generation_confirmed']}\nImages/LLM: ${cost['images_and_llm_estimated']}\nTotal: ${cost['total_estimated']}"
        return final_video_url, cost_text
    except Exception as e:
        return None, f"Generation Failed: {str(e)}"

# --- UI LAYOUT ---
with gr.Blocks(title="AdForge Studio Pro", theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="header-container"):
        gr.Markdown("# ⚡ AdForge Studio Pro")
        gr.Markdown("Enterprise-Grade AI Concept Transfer Engine")
        
        with gr.Row():
            api_key_input = gr.Textbox(
                label="🔑 Secure FAL API Key", 
                type="password", 
                placeholder="Paste your fal_key_... here to authorize generation",
                elem_classes="panel-box"
            )
    
    state_storyboard_frames = gr.State()
    
    with gr.Tabs():
        with gr.TabItem("1. Strategy & Assets"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 📥 Source Material")
                    ad_url = gr.Textbox(label="Reference Ad URL", placeholder="Paste inspiration video link...", show_label=False)
                    btn_download = gr.Button("Fetch Source", variant="secondary")
                    preview_video = gr.Video(label="Source Video", height=300)
                    
                    gr.Markdown("### ⚙️ Production Settings")
                    production_choices = [
                        "✓ Narrative Ad (On-Screen Actor / Dialogue)", 
                        "✓ Narrative Ad (Voiceover focused)", 
                        "x Narrative Ad (No Voiceover - Text/Music focused)", 
                        "x Motion Graphic (SFX & Music focused)"
                    ]
                    production_type = gr.Dropdown(choices=production_choices, value="Narrative Ad (On-Screen Actor / Dialogue)", label="Format Variant")
                    duration_choice = gr.Slider(minimum=3, maximum=60, step=1, value=15, label="Target Duration (sec)")
                    
                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 🎨 Brand Assets & Directives")
                    uploaded_assets = gr.File(label="Asset Hopper (Drop Faces, Logos, UI Screens here)", file_count="multiple", type="filepath", height=150)
                    creative_directive = gr.Textbox(label="Creative Directive", placeholder="E.g., 'Asset 1 is my UI dashboard. Asset 2 is my logo. Swap out the original product for Asset 1.'", lines=4)
                    
                    btn_extract = gr.Button("🧠 Analyze Strategy & Compute Pacing", variant="primary", interactive=False)
                    
            with gr.Row(elem_classes="panel-box"):
                extracted_prompts = gr.Textbox(label="AI Strategy & Dynamic Script (Editable)", lines=12, interactive=True)
                
        with gr.TabItem("2. Storyboard Engine"):
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    aspect_ratio_choice = gr.Dropdown(choices=["16:9", "9:16", "1:1"], value="16:9", label="Aspect Ratio")
                    
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
                    
                    btn_generate_storyboard = gr.Button("🎨 Render Storyboard Matrix", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🔬 Surgical Override")
                    scene_num_input = gr.Number(label="Target Scene Number", precision=0, value=1)
                    btn_regen_single = gr.Button("Re-Render Targeted Frame", variant="secondary")
                    
                with gr.Column(scale=3):
                    storyboard_gallery = gr.Gallery(label="Composited Storyboard", columns=3, height=600)
                    
        with gr.TabItem("3. Final Render"):
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(choices=list(MODELS.keys()), value="gemini-omni-flash", label="Animation Engine")
                    resolution_choice = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    final_instructions = gr.Textbox(label="Global Override Instructions", placeholder="e.g. Keep lighting cinematic, end with logo fade.", lines=3)
                    btn_generate_final = gr.Button("🎬 Render Final Commercial", variant="primary")
                    cost_output = gr.Textbox(label="Compute Cost Estimate", lines=3, interactive=False)
                    
                with gr.Column(scale=3):
                    final_video = gr.Video(label="Final AdForge Export", height=600)

    # WIRING
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
        inputs=[api_key_input, extracted_prompts, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice], 
        outputs=[storyboard_gallery, state_storyboard_frames]
    )
    
    btn_regen_single.click(
        fn=step3_regenerate_single, 
        inputs=[api_key_input, extracted_prompts, scene_num_input, uploaded_assets, preview_video, aspect_ratio_choice, font_style_choice, state_storyboard_frames], 
        outputs=[storyboard_gallery, state_storyboard_frames]
    )
    
    btn_generate_final.click(
        fn=step4_generate_final, 
        inputs=[api_key_input, state_storyboard_frames, final_instructions, model_choice, duration_choice, aspect_ratio_choice, resolution_choice, extracted_prompts, production_type], 
        outputs=[final_video, cost_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)