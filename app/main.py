"""Gradio UI wiring the full AdForge pipeline together — modern dark theme."""
import gradio as gr
from app.pipeline.downloader import download_video
from app.pipeline.uploader import upload_video
from app.pipeline.analyzer import analyze_video
from app.pipeline.generator import generate_video
from app.pipeline.model_registry import MODELS
from app.pipeline.user_assets import process_user_images

CUSTOM_CSS = """
.gradio-container { background: #0d0d12 !important; }
#header { text-align: center; margin-bottom: 10px; }
.gr-button-primary { background: #8b5cf6 !important; border: none !important; }
.panel { border-radius: 16px !important; border: 1px solid #2a2a35 !important; background: #16161d !important; }
"""

THEME = gr.themes.Base(primary_hue="purple", neutral_hue="slate").set(
    body_background_fill="#0d0d12",
    block_background_fill="#16161d",
    block_border_color="#2a2a35",
    button_primary_background_fill="#8b5cf6",
)


def run_pipeline(ad_url, requested_changes, model_choice, duration, aspect_ratio, resolution,
                  user_images, image_descriptions, progress=gr.Progress()):
    progress(0.05, desc="Downloading ad video...")
    local_path = download_video(ad_url)

    progress(0.2, desc="Uploading to fal storage...")
    hosted_url = upload_video(local_path)

    progress(0.35, desc="Processing your reference images...")
    user_image_paths = [f.name for f in user_images] if user_images else []
    user_image_urls, user_image_note = process_user_images(user_image_paths, image_descriptions)

    progress(0.5, desc="Analyzing ad with Nemotron...")
    breakdown = analyze_video(hosted_url, model_choice)

    progress(0.75, desc=f"Generating with {MODELS[model_choice]['label']}...")
    output_path = generate_video(
        model_id=model_choice,
        video_url=hosted_url,
        local_video_path=local_path,
        breakdown=breakdown,
        requested_changes=requested_changes,
        total_duration=duration,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        user_image_urls=user_image_urls,
        user_image_note=user_image_note,
    )

    return breakdown, output_path


with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="AdForge") as demo:
    gr.Markdown("# ⚡ AdForge", elem_id="header")
    gr.Markdown("AI-powered ad recreation pipeline", elem_id="header")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            ad_url = gr.Textbox(label="Reference Ad URL", placeholder="Instagram / TikTok link")
            requested_changes = gr.Textbox(label="Changes wanted", lines=3,
                                            placeholder="e.g. Make this a sad, slow-paced version")

            with gr.Row():
                model_choice = gr.Dropdown(choices=list(MODELS.keys()), value="seedance-2.0-fast", label="Model")
                duration = gr.Slider(minimum=3, maximum=60, step=1, value=8, label="Total Duration (s)")

            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    choices=["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    value="auto", label="Aspect Ratio")
                resolution = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")

            user_images = gr.File(label="Reference images (optional, up to 4)", file_count="multiple",
                                   file_types=["image"])
            image_descriptions = gr.Textbox(
                label="How should each image be used?",
                placeholder="e.g. Image 1 = replace the model's face, Image 2 = use this product instead")

            run_btn = gr.Button("Generate Ad", variant="primary")

        with gr.Column(scale=1, elem_classes="panel"):
            breakdown_output = gr.Textbox(label="Creative Breakdown", lines=14)
            video_output = gr.Video(label="Generated Ad")

    run_btn.click(
        fn=run_pipeline,
        inputs=[ad_url, requested_changes, model_choice, duration, aspect_ratio, resolution,
                user_images, image_descriptions],
        outputs=[breakdown_output, video_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)