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

class BudgetExceeded(Exception):
    """Raised when a projected spend would breach the per-run budget ceiling."""
    pass


class CostTracker:
    def __init__(self, max_budget_usd: float = 0.0):
        self.video_generation_cost = 0.0
        self.analysis_cost = 0.0
        self.fusion_calls = 0
        self.caption_calls = 0
        self.image_generation_cost = 0.0
        # 0 (or falsy) means "no ceiling". Otherwise a hard cap in USD.
        self.max_budget_usd = float(max_budget_usd or 0.0)

    def add_segment(self, model_id: str, duration: int, resolution: str = "720p"):
        self.video_generation_cost += estimate_segment_cost(model_id, duration, resolution)

    def add_analysis_call(self):
        self.analysis_cost += NEMOTRON_ANALYSIS_ESTIMATE

    def add_fusion_call(self):
        self.fusion_calls += 1

    def add_caption_call(self):
        self.caption_calls += 1
        
    def add_image_generation_calls(self, count: int, image_model: str = "nano-banana-2-edit",
                                   image_size: str = "", image_quality: str = "high"):
        """Bill image generation using the real per-model/size/quality price.

        Falls back to the flat nano estimate if the registry can't be consulted,
        so this never crashes a run.
        """
        try:
            from app.new_pipeline.frame_designer import image_price
            per_image = image_price(image_model, image_size, image_quality)
        except Exception:
            per_image = NANO_BANANA_ESTIMATE
        self.image_generation_cost += count * per_image

    def current_total(self) -> float:
        """Live running total of everything billed so far (unrounded)."""
        aux_cost = (self.fusion_calls * FUSION_CALL_ESTIMATE) + (self.caption_calls * CAPTION_CALL_ESTIMATE)
        return (self.video_generation_cost + self.analysis_cost
                + aux_cost + self.image_generation_cost)

    def would_exceed(self, next_cost: float) -> bool:
        """True if spending `next_cost` more would breach the ceiling."""
        if self.max_budget_usd <= 0:
            return False
        return (self.current_total() + next_cost) > self.max_budget_usd

    def check_budget(self, next_cost: float, label: str = "next step"):
        """Raise BudgetExceeded if `next_cost` would breach the ceiling."""
        if self.would_exceed(next_cost):
            raise BudgetExceeded(
                f"Budget cap ${self.max_budget_usd:.2f} would be exceeded by {label}: "
                f"already ${self.current_total():.2f}, {label} adds ${next_cost:.2f}."
            )

    def total(self) -> dict:
        aux_cost = (self.fusion_calls * FUSION_CALL_ESTIMATE) + (self.caption_calls * CAPTION_CALL_ESTIMATE)
        return {
            "video_generation_confirmed": round(self.video_generation_cost, 4),
            "images_and_llm_estimated": round(self.analysis_cost + aux_cost + self.image_generation_cost, 4),
            "total_estimated": round(self.video_generation_cost + self.analysis_cost + aux_cost + self.image_generation_cost, 4)
        }