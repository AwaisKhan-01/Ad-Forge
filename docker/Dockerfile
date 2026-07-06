FROM python:3.11-slim

# ffmpeg needed by yt-dlp for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# No FAL_KEY here — it's injected at runtime by each user, never baked in
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app/main.py"]
