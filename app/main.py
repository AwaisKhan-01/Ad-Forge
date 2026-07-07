"""Gradio UI wiring the full AdForge pipeline together."""
import gradio as gr
from app.pipeline.downloader import download_video
from app.pipeline.uploader import upload_video
from app.pipeline.analyzer import analyze_video
from app.pipeline.generator import generate_video


def run_pipeline(ad_url: str, requested_changes: str, progress=gr.Progress()):
    progress(0.1, desc="Downloading ad video...")
    local_path = download_video(ad_url)

    progress(0.3, desc="Uploading to fal storage...")
    hosted_url = upload_video(local_path)

    progress(0.5, desc="Analyzing ad with Nemotron...")
    breakdown = analyze_video(hosted_url)

    progress(0.8, desc="Generating new video with Seedance...")
    output_url = generate_video(hosted_url, breakdown, requested_changes)

    return breakdown, output_url


with gr.Blocks() as demo:
    gr.Markdown("# AdForge — AI Ad Recreation Pipeline")

    ad_url = gr.Textbox(label="Reference Ad URL (Instagram/TikTok link)")
    requested_changes = gr.Textbox(
        label="What changes do you want?",
        placeholder="e.g. Change product to a red sneaker, swap background to a beach"
    )
    run_btn = gr.Button("Generate")

    breakdown_output = gr.Textbox(label="Creative Breakdown (from Nemotron)", lines=10)
    video_output = gr.Video(label="Generated Ad")

    run_btn.click(
        fn=run_pipeline,
        inputs=[ad_url, requested_changes],
        outputs=[breakdown_output, video_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)