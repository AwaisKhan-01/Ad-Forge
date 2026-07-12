"""
Parses structured text, extracts Dead-Center Keyframes, and composites them in PARALLEL.
Features the Diffusion Prompt Sterilizer with a Native Python NLP Fallback.
Located in: app/new_pipeline/frame_designer.py
"""
import os
import re
import uuid
import subprocess
import fal_client
import concurrent.futures
from typing import List, Dict
from app.pipeline.uploader import upload_video

IMAGE_EDIT_MODEL = "fal-ai/nano-banana-2/edit"
TEXT_LLM_MODEL = "fal-ai/llama-3.1-8b-instruct"
FRAMES_DIR = "frames"

# ==========================================
# 1. DEAD-CENTER EXTRACTION MATH
# ==========================================
def parse_timestamp_to_center_seconds(ts_str: str) -> float:
    try:
        parts = ts_str.replace(" ", "").split('-')
        def time_to_sec(t_str):
            if ':' in t_str:
                m, s = t_str.split(':')
                return int(m) * 60 + float(s)
            return float(t_str)
        
        start_sec = time_to_sec(parts[0])
        if len(parts) > 1:
            end_sec = time_to_sec(parts[1])
            duration = end_sec - start_sec
            return start_sec + (duration / 2.0)
        
        return start_sec + 0.5
    except Exception as e:
        print(f"Timestamp parsing error: {e}")
        return 0.0

def extract_frame(video_path: str, timestamp_sec: float) -> str:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    out_path = os.path.join(FRAMES_DIR, f"{uuid.uuid4()}.jpg")
    cmd = [
        "ffmpeg", "-ss", str(timestamp_sec), "-i", video_path,
        "-frames:v", "1", "-q:v", "2", out_path, "-y"
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return out_path if os.path.exists(out_path) else None

# ==========================================
# 2. THE DIFFUSION PROMPT STERILIZER
# ==========================================
def _compile_diffusion_prompt(visual_text: str, is_text_banned: bool) -> str:
    ban_instruction = "CRITICAL: Omit any tags related to text, typography, or writing." if is_text_banned else "Include exact text strings in quotes if specified."
    
    compiler_prompt = f"""You are a Stable Diffusion Prompt Engineer. Translate this scene into a highly optimized, comma-separated list of visual tags.

SCENE DESCRIPTION: "{visual_text}"

RULES:
1. Strip out ALL conversational words, verbs, and meta-instructions (e.g., "Transform this", "The camera shows", "appears").
2. Output ONLY visual keywords.
3. {ban_instruction}
4. Ensure UI elements are described flatly.

OUTPUT FORMAT: Return ONLY the comma-separated string.
"""
    try:
        # THE FIX: Using a lightning-fast Text LLM instead of the heavy Video VLM
        result = fal_client.subscribe(
            TEXT_LLM_MODEL, 
            arguments={
                "prompt": compiler_prompt, 
                "system_prompt": "You are a compiler. Output only comma-separated tags.", 
                "max_tokens": 100, 
                "temperature": 0.1
            }
        )
        compiled_tags = result.get("output", "").strip()
        print(f"      [STERILIZED TAGS (LLM)] -> {compiled_tags}")
        return compiled_tags
        
    except Exception as e:
        print(f"      [SYSTEM WARNING] LLM Compiler failed ({e}). Engaging Native Python NLP fallback.")
        # THE FIX: Bulletproof local text parsing if the API fails
        quoted_texts = re.findall(r"'(.*?)'|\"(.*?)\"", visual_text)
        quotes = [q[0] or q[1] for q in quoted_texts if q[0] or q[1]]
        
        clean = visual_text.lower()
        bad = ["shows", "appears", "transform", "camera", "scene", "visual", "displaying", "split screen", "top", "bottom"]
        for b in bad: clean = clean.replace(b, "")
        
        # Keep letters, numbers, and spaces
        clean = re.sub(r'[^a-z0-9\s]', ' ', clean)
        tags = [t.strip() for t in clean.split() if len(t.strip()) > 2]
        
        # Deduplicate and format
        final_tags = ", ".join(list(dict.fromkeys(tags)))
        
        if not is_text_banned and quotes:
            final_tags += ", " + ", ".join([f"text '{q}'" for q in quotes])
            
        print(f"      [STERILIZED TAGS (LOCAL)] -> {final_tags}")
        return final_tags

# ==========================================
# 3. PARSING & ORCHESTRATION
# ==========================================
def parse_extracted_scenes(extracted_text: str) -> List[Dict[str, str]]:
    scenes = []
    pattern = re.compile(
        r"SCENE\s+(\d+)\s*"
        r"Timestamp:\s*(.*?)\s*"
        r"Visual:\s*(.*?)\s*"
        r"Motion:\s*(.*?)\s*"
        r"Audio:\s*(.*?)(?=\nSCENE|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    
    matches = pattern.finditer(extracted_text)
    for match in matches:
        scenes.append({
            "scene_number": match.group(1).strip(),
            "timestamp": match.group(2).strip(),
            "visual_prompt": match.group(3).strip(),
            "motion": match.group(4).strip(),
            "audio": match.group(5).strip()
        })
    return scenes

def _process_single_scene(scene: Dict, asset_urls: List[str], local_video_path: str, aspect_ratio: str) -> Dict:
    target_time = parse_timestamp_to_center_seconds(scene['timestamp'])
    print(f"\n[EXTRACTING] Scene {scene.get('scene_number')} -> Centered Timecode: {target_time:.2f}s")
    
    frame_path = extract_frame(local_video_path, target_time)
    
    if not frame_path:
        scene["generated_image_url"] = None
        return scene
        
    base_frame_url = upload_video(frame_path)
    image_urls = [base_frame_url]
    
    raw_visual_text = scene['visual_prompt']
    
    trigger_words = ["text", "word", "typography", "font", "color", "red", "pink", "blue", "yellow", "white", "black", "interface", "notification"]
    is_text_banned = not any(word in raw_visual_text.lower() for word in trigger_words)
    
    if not asset_urls:
        raw_visual_text = re.sub(r'#Image\d+', '', raw_visual_text, flags=re.IGNORECASE).strip()

    sterilized_tags = _compile_diffusion_prompt(raw_visual_text, is_text_banned)
    
    design_prompt = f"{sterilized_tags}, perfect layout matching #Image1, highly detailed"
    
    if asset_urls:
        for idx, url in enumerate(asset_urls):
            asset_tag = f"#Image{idx + 2}" 
            image_urls.append(url)
            if asset_tag.lower() in raw_visual_text.lower():
                design_prompt += f", replace main subject with {asset_tag}"
            else:
                design_prompt += f", morph human likeness to match {asset_tag}"
    
    args = {"prompt": design_prompt, "image_urls": image_urls, "aspect_ratio": aspect_ratio}
    
    try:
        result = fal_client.subscribe(IMAGE_EDIT_MODEL, arguments=args)
        scene["generated_image_url"] = result["images"][0]["url"] if result and "images" in result else None
    except Exception as e:
        print(f"Error on Scene {scene['scene_number']}: {e}")
        scene["generated_image_url"] = None 
        
    scene["visual_prompt_sterilized"] = sterilized_tags
    return scene

def generate_storyboard_frames(extracted_text: str, uploaded_assets: List[str], local_video_path: str, aspect_ratio: str = "9:16") -> List[Dict]:
    scenes = parse_extracted_scenes(extracted_text)
    if not scenes:
        raise ValueError("Could not parse scenes. Check VLM Agent output format.")
        
    asset_urls = []
    if uploaded_assets:
        for asset in uploaded_assets:
            if asset:
                url = upload_video(asset) if not asset.startswith("http") else asset
                asset_urls.append(url)
            
    storyboard_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for scene in scenes:
            futures.append(executor.submit(_process_single_scene, scene, asset_urls, local_video_path, aspect_ratio))
            
        for future in concurrent.futures.as_completed(futures):
            storyboard_results.append(future.result())
            
    storyboard_results.sort(key=lambda x: int(x["scene_number"]))
    return storyboard_results