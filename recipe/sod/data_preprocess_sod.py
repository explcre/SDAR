#!/usr/bin/env python
# Copyright 2026 — SOD benchmark integration for SDAR.
# Licensed under the Apache License, Version 2.0.
"""Convert the SOD AIME eval set (Open-AgentRL-Eval: AIME2024 + AIME2025) into SDAR/verl
multi-turn TIR (tool-integrated reasoning) format for the SOD benchmark.

- Reconstructs SOD's *exact* AIME prompt (from the SOD repo recipe/demystify/reward.py),
  so the benchmark is faithful to the released SOD evaluation.
- Wires the `code_interpreter` tool (verl.tools.sandbox_fusion_tools) via
  extra_info.tools_kwargs, so the model can execute Python during reasoning (TIR).
- Sets data_source='aime2024'/'aime2025' so verl's reward dispatcher routes scoring to
  `math_dapo` (strict boxed-answer match) — the SAME scorer SOD uses.

Usage:
  python recipe/sod/data_preprocess_sod.py \
      --src_dir /path/to/Open-AgentRL-Eval --out_dir ~/data/sod_aime
Produces <out_dir>/test.parquet (AIME2024 + AIME2025 concatenated, 60 problems).
"""
import argparse
import glob
import os

import pandas as pd

# --- SOD's exact AIME TIR prompt pieces (verbatim from SOD recipe/demystify/reward.py) ---
MATH_P1 = "Analyze and solve the following math problem step by step. \n\n"
MATH_P2 = (
    "\n\nThe tool could be used for more precise and efficient calculation and could "
    "help you to verify your result before you reach the final answer."
)
AGENT_P = (
    "\n\n**Note: You should first analyze the problem and form a high-level solution "
    "strategy, then utilize the tools to help you solve the problem.**"
)
ANSWER_FMT = (
    "\nRemember once you make sure the current answer is your final answer, do not call "
    "the tools again and directly output the final answer in the following text format, "
    "the answer format must be: \\boxed{'The final answer goes here.'}."
)
def build_prompt(problem: str) -> str:
    """Reconstruct SOD's AIME tool-integrated-reasoning user prompt for one problem.

    Matches SOD recipe/demystify/reward.py map_fn for AIME *exactly*:
    math_prompt_1 + problem + math_prompt_2 + agent_prompt + answer_format.
    (SOD's no-units suffix is only in map_fn2/skywork, NOT the AIME path — so it is omitted here.)
    """
    return MATH_P1 + problem + MATH_P2 + AGENT_P + ANSWER_FMT


def convert(src_dir: str, out_dir: str) -> None:
    """Read the two SOD AIME eval parquets and emit one SDAR/verl multi-turn test parquet."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sub, data_source in (("aime2024", "aime2024"), ("aime2025", "aime2025")):
        matches = glob.glob(os.path.join(src_dir, sub, "*.parquet"))
        if not matches:
            raise FileNotFoundError(f"no parquet under {os.path.join(src_dir, sub)}")
        df = pd.read_parquet(matches[0])
        # the two sources differ in column casing (AIME2024: Problem/Answer; AIME2025: problem/answer)
        pcol = "Problem" if "Problem" in df.columns else "problem"
        acol = "Answer" if "Answer" in df.columns else "answer"
        for idx, r in df.reset_index(drop=True).iterrows():
            gt = str(r[acol]).strip()
            rows.append({
                "data_source": data_source,
                "prompt": [{"role": "user", "content": build_prompt(str(r[pcol]))}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": "test",
                    "index": int(idx),
                    "data_source": data_source,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "code_interpreter": {"create_kwargs": {"ground_truth": gt}},
                    },
                },
            })
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "test.parquet")
    out.to_parquet(path, index=False)
    n24 = int((out["data_source"] == "aime2024").sum())
    n25 = int((out["data_source"] == "aime2025").sum())
    print(f"wrote {len(out)} rows -> {path}  (aime2024={n24}, aime2025={n25})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="dir containing aime2024/ and aime2025/ parquets (SOD Open-AgentRL-Eval)")
    ap.add_argument("--out_dir", default=os.path.expanduser("~/data/sod_aime"))
    args = ap.parse_args()
    convert(args.src_dir, args.out_dir)
