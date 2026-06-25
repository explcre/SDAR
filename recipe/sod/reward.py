#!/usr/bin/env python
# Copyright 2026 — SOD benchmark integration for SDAR. Apache-2.0.
"""SOD benchmark reward: strict boxed-answer scoring for AIME-TIR.

Reuses verl's `math_dapo` scorer (the SAME one SOD uses) but forces
`strict_box_verify=True`. This matters: verl's default reward dispatcher calls
`math_dapo.compute_score(...)` with `strict_box_verify=False`, which uses Minerva-style
"Answer:" regex matching — but SOD's prompt instructs the model to box the final answer
(`\\boxed{...}`), so SOD grades with STRICT BOX extraction. Wiring this as a
custom_reward_function keeps the shared dispatcher untouched.

Wire via: custom_reward_function.path=recipe/sod/reward.py custom_reward_function.name=compute_score
The verl reward manager calls compute_score(data_source=, solution_str=, ground_truth=, extra_info=).
"""
from verl.utils.reward_score import math_dapo


def compute_score(data_source=None, solution_str: str = "", ground_truth: str = "", extra_info=None, **kwargs):
    """Strict-box AIME score via math_dapo. Returns dict {score, acc, pred} (verl handles it)."""
    return math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
