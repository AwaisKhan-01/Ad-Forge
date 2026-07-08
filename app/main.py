"""Gradio UI wiring the full AdForge pipeline together — modern dark theme, direct @ImageN prompting."""
import gradio as gr
from app.pipeline.downloader import download_video
from app.pipeline.uploader import upload_video
from app.pipeline.analyzer import analyze_video
from app.pipeline.generator import generate_video
from app.pipeline.model_registry import MODELS
from app.pipeline.user_assets import process_user_images
from app.pipeline.cost_tracker import CostTracker

CUSTOM_CSS = """
.gradio-container { background: #0d0d12 !important; }
#header { text-align: center; margin-bottom: 10px; }
.gr-button-primary { background: #8b5cf6 !important; border: none !important; }
.panel { border-radius: 16px !important; border: 1px solid #2a2a35 !important; background: #16161d !important; }
.image-tag { font-size: 0.8em; color: #8b5cf6; text-align: center; margin-top: -8px; }
"""

THEME = gr.themes.Base(primary_hue="purple", neutral_hue="slate").set(
    body_background_fill="#0d0d12",
    block_background_fill="#16161d",
    block_border_color="#2a2a35",
    button_primary_background_fill="#8b5cf6",
)


def preview_uploaded_images(files):
    """Shows uploaded images labeled with their @ImageN tag so the user knows what to reference in the prompt."""
    if not files:
        return []
    return [(f.name, f"@Image{i+1}") for i, f in enumerate(files)]


def run_pipeline(ad_url, prompt, model_choice, duration, aspect_ratio, resolution,
                  user_images, progress=gr.Progress()):
    tracker = CostTracker()

    progress(0.05, desc="Downloading ad video...")
    local_path = download_video(ad_url)

    progress(0.2, desc="Uploading to fal storage...")
    hosted_url = upload_video(local_path)

    progress(0.35, desc="Processing your reference images...")
    user_image_paths = [f.name for f in user_images] if user_images else []
    user_image_urls = process_user_images(user_image_paths)

    progress(0.5, desc="Analyzing ad with Nemotron...")
    breakdown = analyze_video(hosted_url, model_choice)
    tracker.add_analysis_call()

    progress(0.75, desc=f"Generating with {MODELS[model_choice]['label']}...")
    output_path = generate_video(
        model_id=model_choice,
        video_url=hosted_url,
        local_video_path=local_path,
        breakdown=breakdown,
        requested_changes=prompt,  # user's @ImageN-referencing prompt goes straight through
        total_duration=duration,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        user_image_urls=user_image_urls,
        cost_tracker=tracker,
    )

    cost = tracker.total()
    cost_text = (
        f"Video generation (confirmed): ${cost['video_generation_confirmed']}\n"
        f"Analysis/LLM calls (estimated): ${cost['analysis_and_llm_estimated']}\n"
        f"Total estimated cost: ${cost['total_estimated']}"
    )

    return breakdown, output_path, cost_text


def update_duration_range(model_choice):
    d = MODELS[model_choice]["durations"]
    return gr.update(minimum=min(d), maximum=max(d), value=min(max(d), 8))


with gr.Blocks(title="AdForge") as demo:
    gr.Markdown("# ⚡ AdForge", elem_id="header")
    gr.Markdown("AI-powered ad recreation pipeline", elem_id="header")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            ad_url = gr.Textbox(label="Reference Ad URL", placeholder="Instagram / TikTok link")

            gr.Markdown("**Reference images** (optional) — uploaded images are tagged in order below. Reference them in your prompt as `@Image1`, `@Image2`, etc.")
            user_images = gr.File(label="Upload images", file_count="multiple", file_types=["image"])
            image_gallery = gr.Gallery(label="Tagged Images", columns=4, height=140, show_label=True)

            prompt = gr.Textbox(
                label="Your instructions",
                lines=4,
                placeholder="e.g. Use the person from @Image1 as the subject, use @Image2 as the UI the animation runs on, place @Image3 as the logo in the corner."
            )

            with gr.Row():
                model_choice = gr.Dropdown(choices=list(MODELS.keys()), value="seedance-2.0-fast", label="Model")
                duration = gr.Slider(minimum=3, maximum=60, step=1, value=8, label="Total Duration (s)")

            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    choices=["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    value="auto", label="Aspect Ratio")
                resolution = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")

            run_btn = gr.Button("Generate Ad", variant="primary")

        with gr.Column(scale=1, elem_classes="panel"):
            breakdown_output = gr.Textbox(label="Creative Breakdown", lines=10)
            video_output = gr.Video(label="Generated Ad")
            cost_output = gr.Textbox(label="Estimated Cost", lines=3)

    user_images.change(fn=preview_uploaded_images, inputs=[user_images], outputs=[image_gallery])

    model_choice.change(fn=update_duration_range, inputs=[model_choice], outputs=[duration])

    run_btn.click(
        fn=run_pipeline,
        inputs=[ad_url, prompt, model_choice, duration, aspect_ratio, resolution, user_images],
        outputs=[breakdown_output, video_output, cost_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CUSTOM_CSS)