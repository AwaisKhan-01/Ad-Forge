# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies (FFmpeg is CRITICAL for your frame extraction and stitching)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code securely (NO .env FILE)
COPY app/ ./app/

# Ensure the dynamically generated directories exist so the container doesn't crash on write
RUN mkdir -p frames segments outputs downloads

# Gradio serves on 7860
EXPOSE 7860

# Run the main entrypoint module
CMD ["python", "-m", "app.main"]