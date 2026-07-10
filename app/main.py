"""
Gradio UI for AdForge: Enterprise Dashboard.
Features a tabbed workflow, professional CSS, and decoupled logic.
"""
import gradio as gr
from app.pipeline.downloader import download_video
from app.pipeline.uploader import upload_video
from app.pipeline.model_registry import MODELS
from app.pipeline.cost_tracker import CostTracker

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

def step1_download(url):
    local_path = download_video(url)
    return local_path, gr.update(interactive=True)

def step2_extract_scenes(local_video_path, creative_directive, target_duration, production_type):
    """Now purely passes user intent to the backend. No hardcoded math."""
    if not local_video_path:
        return "Error: Please download a video first."
    try:
        hosted_video_url = upload_video(local_video_path)
        # Note: exact_scenes math is removed. The analyzer handles it now.
        extracted_text = analyze_and_extract_scenes(
            video_url=hosted_video_url, 
            target_duration=int(target_duration),
            creative_directive=creative_directive,
            production_type=production_type
        )
        return extracted_text
    except Exception as e:
        return f"PIPELINE ERROR: {str(e)}"

def step3_generate_storyboard(edited_prompts, uploaded_assets, local_video_path, aspect_ratio):
    if not edited_prompts:
        return [], None
    asset_paths = uploaded_assets if uploaded_assets else []
    try:
        storyboard_data = generate_storyboard_frames(edited_prompts, asset_paths, local_video_path, aspect_ratio)
        gallery_format = [(frame["generated_image_url"], f"Scene {frame['scene_number']}: {frame.get('audio', frame.get('sfx', ''))}") for frame in storyboard_data if frame.get("generated_image_url")]
        return gallery_format, storyboard_data
    except Exception as e:
        print(f"Storyboard compilation error: {str(e)}")
        return [], None

def step3_regenerate_single(edited_prompts, scene_num_to_regen, uploaded_assets, local_video_path, aspect_ratio, current_storyboard_state):
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
                
        updated_scene = _process_single_scene(target_scene_data, asset_urls, local_video_path, aspect_ratio)
        
        for idx, old_scene in enumerate(current_storyboard_state):
            if str(old_scene["scene_number"]) == str(scene_num_to_regen):
                current_storyboard_state[idx] = updated_scene
                break
                
        gallery_format = [(frame["generated_image_url"], f"Scene {frame['scene_number']}: {frame.get('audio', frame.get('sfx', ''))}") for frame in current_storyboard_state if frame.get("generated_image_url")]
        gr.Info(f"Scene {scene_num_to_regen} successfully updated!")
        return gallery_format, current_storyboard_state
    except Exception as e:
        print(f"Single scene regeneration crash: {e}")
        return gr.update(), current_storyboard_state

def step4_generate_final(storyboard_frames, final_instructions, model_choice, duration, aspect_ratio, resolution, raw_script_text, production_type):
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
    
    state_storyboard_frames = gr.State()
    
    with gr.Tabs():
        # TAB 1: SETUP & ASSETS
        with gr.TabItem("1. Strategy & Assets"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 📥 Source Material")
                    ad_url = gr.Textbox(label="Reference Ad URL", placeholder="Paste inspiration video link...", show_label=False)
                    btn_download = gr.Button("Fetch Source", variant="secondary")
                    preview_video = gr.Video(label="Source Video", height=300)
                    
                    gr.Markdown("### ⚙️ Production Settings")
                    production_type = gr.Dropdown(choices=["Narrative Ad (Voiceover focused)", "Narrative Ad (No Voiceover - Text/Music focused)", "Motion Graphic (SFX & Music focused)"], value="Narrative Ad (Voiceover focused)", label="Format Variant")
                    duration_choice = gr.Slider(minimum=3, maximum=60, step=1, value=15, label="Target Duration (sec)")
                    
                with gr.Column(scale=1, elem_classes="panel-box"):
                    gr.Markdown("### 🎨 Brand Assets & Directives")
                    uploaded_assets = gr.File(label="Asset Hopper (Drop Faces, Logos, UI Screens here)", file_count="multiple", type="filepath", height=150)
                    creative_directive = gr.Textbox(label="Creative Directive", placeholder="E.g., 'Asset 1 is my UI dashboard. Asset 2 is my logo. Swap out the original product for Asset 1.'", lines=4)
                    
                    btn_extract = gr.Button("🧠 Analyze Strategy & Compute Pacing", variant="primary", interactive=False)
                    
            with gr.Row(elem_classes="panel-box"):
                extracted_prompts = gr.Textbox(label="AI Strategy & Dynamic Script (Editable)", lines=12, interactive=True)
                
        # TAB 2: STORYBOARD ENGINE
        with gr.TabItem("2. Storyboard Engine"):
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    aspect_ratio_choice = gr.Dropdown(choices=["16:9", "9:16", "1:1"], value="16:9", label="Aspect Ratio")
                    btn_generate_storyboard = gr.Button("🎨 Render Storyboard Matrix", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🔬 Surgical Override")
                    scene_num_input = gr.Number(label="Target Scene Number", precision=0, value=1)
                    btn_regen_single = gr.Button("Re-Render Targeted Frame", variant="secondary")
                    
                with gr.Column(scale=3):
                    storyboard_gallery = gr.Gallery(label="Composited Storyboard", columns=3, height=600)
                    
        # TAB 3: FINAL RENDER
        with gr.TabItem("3. Final Render"):
            with gr.Row(elem_classes="panel-box"):
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(choices=list(MODELS.keys()), value="wan-2.2", label="Animation Engine")
                    resolution_choice = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    final_instructions = gr.Textbox(label="Global Override Instructions", placeholder="e.g. Keep lighting cinematic, end with logo fade.", lines=3)
                    btn_generate_final = gr.Button("🎬 Render Final Commercial", variant="primary")
                    cost_output = gr.Textbox(label="Compute Cost Estimate", lines=3, interactive=False)
                    
                with gr.Column(scale=3):
                    final_video = gr.Video(label="Final AdForge Export", height=600)

    # WIRING
    btn_download.click(fn=step1_download, inputs=[ad_url], outputs=[preview_video, btn_extract])
    
    # Notice: We no longer pass exact_scenes or model_choice to Pass 1
    btn_extract.click(fn=step2_extract_scenes, inputs=[preview_video, creative_directive, duration_choice, production_type], outputs=[extracted_prompts])
    
    btn_generate_storyboard.click(fn=step3_generate_storyboard, inputs=[extracted_prompts, uploaded_assets, preview_video, aspect_ratio_choice], outputs=[storyboard_gallery, state_storyboard_frames])
    
    btn_regen_single.click(fn=step3_regenerate_single, inputs=[extracted_prompts, scene_num_input, uploaded_assets, preview_video, aspect_ratio_choice, state_storyboard_frames], outputs=[storyboard_gallery, state_storyboard_frames])
    
    btn_generate_final.click(fn=step4_generate_final, inputs=[state_storyboard_frames, final_instructions, model_choice, duration_choice, aspect_ratio_choice, resolution_choice, extracted_prompts, production_type], outputs=[final_video, cost_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)