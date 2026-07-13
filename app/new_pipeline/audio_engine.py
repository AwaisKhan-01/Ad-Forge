"""
Generates TTS voiceover using F5-TTS (with a default cloned voice) 
and micro-scales it to the Master Clock.
Located in: app/new_pipeline/audio_engine.py
"""
import os
import re
import uuid
import requests
import subprocess
import fal_client

AUDIO_DIR = "outputs/audio"
TTS_MODEL = "fal-ai/f5-tts" 

def generate_and_scale_tts(raw_script_text: str, target_duration: int) -> str:
    """
    Extracts the voiceover, generates a cloned TTS audio track, 
    and applies an atempo filter to match the target duration perfectly.
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)
    raw_audio_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}_raw.mp3")
    scaled_audio_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}_master.mp3")
    
    # 1. Parse Voiceover from the UI Textbox
    match = re.search(r"Voiceover:\s*(.*?)(?=\nMusic:|\n---|\nSCENE|\Z)", raw_script_text, re.DOTALL | re.IGNORECASE)
    voiceover_text = match.group(1).strip() if match else "Experience the power of our new product today."
    
    print(f"\nGenerating F5-TTS for: '{voiceover_text[:50]}...'")

    # 2. F5-TTS Payload
    # FIXED: F5-TTS strictly requires a reference audio file to clone and the model_type.
    result = fal_client.subscribe(
        TTS_MODEL,
        arguments={
            "gen_text": voiceover_text,
            "model_type": "F5-TTS",
            # This is a standard clear voice sample provided by Fal.ai for general use
            "ref_audio_url": "https://storage.googleapis.com/falserverless/example_inputs/reference_audio.wav" 
        }
    )
    
    audio_url = result.get("audio", {}).get("url")
    if not audio_url: 
        raise RuntimeError("F5-TTS failed to return an audio URL.")
        
    # Download raw audio
    try:
        r = requests.get(audio_url, timeout=60)
        r.raise_for_status()
        with open(raw_audio_path, "wb") as f: 
            f.write(r.content)
    except Exception as e:
        raise RuntimeError(f"Failed to download TTS audio: {e}")
        
    # 3. Probe and Scale
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
         "-of", "default=noprint_wrappers=1:nokey=1", raw_audio_path], 
        capture_output=True, text=True
    )
    
    try:
        original_duration = float(probe.stdout.strip())
    except ValueError:
        original_duration = float(target_duration)
    
    # Gently stretch or squeeze the audio to fit the Master Clock
    tempo_ratio = max(0.5, min(original_duration / target_duration, 100.0))
    
    cmd = [
        "ffmpeg", "-i", raw_audio_path, 
        "-filter:a", f"atempo={tempo_ratio}", 
        "-t", str(target_duration), 
        "-y", scaled_audio_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    
    return scaled_audio_path