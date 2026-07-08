"""Splits a target ad duration into segments that respect a model's per-call duration limits."""
import math


def plan_segments(total_duration: int, model: dict) -> list[int]:
    """Returns a list of per-segment durations summing to ~total_duration."""
    max_seg = max(model["durations"])
    min_seg = min(model["durations"])

    if total_duration <= max_seg:
        return [max(total_duration, min_seg)]

    num_segments = math.ceil(total_duration / max_seg)
    base = total_duration // num_segments
    remainder = total_duration % num_segments

    segments = [base + (1 if i < remainder else 0) for i in range(num_segments)]
    return [max(s, min_seg) for s in segments]