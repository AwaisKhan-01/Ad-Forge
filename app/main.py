"""Gradio UI wiring the full AdForge pipeline together."""
import gradio as gr
from app.pipeline.downloader import download_video
from app.pipeline.uploader import upload_video
from app.pipeline.analyzer import analyze_video
from app.pipeline.generator import generate_video
from app.pipeline.model_registry import MODELS


def run_pipeline(ad_url, requested_changes, model_choice, duration, aspect_ratio, resolution,
                  progress=gr.Progress()):
    progress(0.1, desc="Downloading ad video...")
    local_path = download_video(ad_url)

    progress(0.3, desc="Uploading to fal storage...")
    hosted_url = upload_video(local_path)

    progress(0.5, desc="Analyzing ad with Nemotron...")
    breakdown = analyze_video(hosted_url)

    progress(0.8, desc=f"Generating with {MODELS[model_choice]['label']}...")
    output_url = generate_video(
        model_id=model_choice,
        video_url=hosted_url,
        local_video_path=local_path,
        breakdown=breakdown,
        requested_changes=requested_changes,
        duration=duration,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )

    return breakdown, output_url


with gr.Blocks() as demo:
    gr.Markdown("# AdForge — AI Ad Recreation Pipeline")

    ad_url = gr.Textbox(label="Reference Ad URL (Instagram/TikTok link)")
    requested_changes = gr.Textbox(
        label="What changes do you want?",
        placeholder="e.g. Change product to a red sneaker, swap background to a beach"
    )

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=list(MODELS.keys()),
            value="seedance-2.0-fast",
            label="Generation Model",
        )
        duration = gr.Slider(minimum=3, maximum=15, step=1, value=8, label="Duration (seconds)")

    with gr.Row():
        aspect_ratio = gr.Dropdown(
            choices=["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
            value="auto",
            label="Aspect Ratio",
        )
        resolution = gr.Dropdown(
            choices=["480p", "720p"],
            value="720p",
            label="Resolution (Seedance only)",
        )

    run_btn = gr.Button("Generate")

    breakdown_output = gr.Textbox(label="Creative Breakdown (from Nemotron)", lines=10)
    video_output = gr.Video(label="Generated Ad")

    run_btn.click(
        fn=run_pipeline,
        inputs=[ad_url, requested_changes, model_choice, duration, aspect_ratio, resolution],
        outputs=[breakdown_output, video_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)