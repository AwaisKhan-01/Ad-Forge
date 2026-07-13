"""Estimates the cost of a full pipeline run, including Nano Banana 2 Edit."""
# CONFIRMED from fal.ai pricing
SEEDANCE_FAST_720P_PER_SEC = 0.2419
SEEDANCE_VIDEO_INPUT_MULTIPLIER = 0.6 

# ESTIMATED
OMNI_PER_SEC_ESTIMATE = 0.10
NEMOTRON_ANALYSIS_ESTIMATE = 0.02
FUSION_CALL_ESTIMATE = 0.01
CAPTION_CALL_ESTIMATE = 0.005
NANO_BANANA_ESTIMATE = 0.015  # Added for Nano Banana 2 Edit

def estimate_segment_cost(model_id: str, duration: int, resolution: str = "720p"):
    if model_id == "seedance-2.0-fast":
        rate = SEEDANCE_FAST_720P_PER_SEC if resolution == "720p" else SEEDANCE_FAST_720P_PER_SEC
        return duration * rate * SEEDANCE_VIDEO_INPUT_MULTIPLIER
    elif model_id == "gemini-omni-flash":
        return duration * OMNI_PER_SEC_ESTIMATE
    return 0.0

class CostTracker:
    def __init__(self):
        self.video_generation_cost = 0.0
        self.analysis_cost = 0.0
        self.fusion_calls = 0
        self.caption_calls = 0
        self.image_generation_cost = 0.0

    def add_segment(self, model_id: str, duration: int, resolution: str = "720p"):
        self.video_generation_cost += estimate_segment_cost(model_id, duration, resolution)

    def add_analysis_call(self):
        self.analysis_cost += NEMOTRON_ANALYSIS_ESTIMATE

    def add_fusion_call(self):
        self.fusion_calls += 1

    def add_caption_call(self):
        self.caption_calls += 1
        
    def add_image_generation_calls(self, count: int):
        self.image_generation_cost += count * NANO_BANANA_ESTIMATE

    def total(self) -> dict:
        aux_cost = (self.fusion_calls * FUSION_CALL_ESTIMATE) + (self.caption_calls * CAPTION_CALL_ESTIMATE)
        return {
            "video_generation_confirmed": round(self.video_generation_cost, 4),
            "images_and_llm_estimated": round(self.analysis_cost + aux_cost + self.image_generation_cost, 4),
            "total_estimated": round(self.video_generation_cost + self.analysis_cost + aux_cost + self.image_generation_cost, 4)
        }