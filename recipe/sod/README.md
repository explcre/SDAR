# SOD benchmark (AIME tool-integrated reasoning) for SDAR

Adds the **SOD** evaluation — AIME 2024/2025 math solved with **tool-integrated reasoning**
(the model writes Python that a SandboxFusion code-interpreter executes), scored **avg@32**
with **strict boxed-answer** matching — as a benchmark in SDAR. Faithful to the released SOD
eval (`youngzhong/SOD-1.7B`, repo `YoungZ365/SOD` recipe/demystify).

## Status (be precise)

**Built + unit-tested (correct in isolation, reusing SDAR infra):**
- `data_preprocess_sod.py` — converts SOD's Open-AgentRL-Eval AIME2024+2025 (60 problems) to
  SDAR/verl multi-turn format. Reproduces SOD's **exact** AIME prompt and wires the
  `code_interpreter` tool via `extra_info.tools_kwargs`. (validated: 60 rows, 30+30)
- `reward.py` — strict-box AIME scorer = `verl math_dapo` with `strict_box_verify=True`
  (SOD's scoring; verl's default dispatcher uses Minerva "Answer:" matching, which is wrong here).
  (unit-tested: correct→+1, wrong→−1, no-box→−1)
- `test_sod_scorer.py` — CPU unit test for the reward path (passes).
- `../../examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml` —
  `code_interpreter` tool backed by `verl.tools.sandbox_fusion_tools.SandboxFusionTool`.

**NOT yet runnable end-to-end (architecture finding from the audit):**
SDAR's trainers (`verl/trainer/main_ppo.py`, `recipe/dapo`, …) route **all** multi-turn
rollout — *including tool use* — through an **`agent_system` environment**:
`_validate()` (verl/trainer/ppo/ray_trainer.py:749) generates via
`self.traj_collector.multi_turn_loop(envs=…)`, and `multi_turn_loop`
(agent_system/multi_turn_rollout/rollout_loop.py:304) **requires** an `EnvironmentManagerBase`
(`envs.reset()/step()`). There is **no stock-verl pure-tool eval path** in this fork (the
sglang tool configs the e2e scripts reference don't exist; `reward_manager` outside `episode`
is rejected by `main_ppo`). So `run_sod_eval.sh` as written (stock-verl flags) will not run here.

## To make it run faithfully: add an AIME-TIR environment

The SDAR-native way is a new `agent_system` environment, e.g. `MathToolEnvironmentManager`:
- `reset()` → present the AIME problem (the prompt above) as the first observation.
- `step(text_actions)` → detect `code_interpreter` tool calls, execute via the existing
  `SandboxFusionTool`, return stdout as the next observation; terminate on a boxed answer or
  `max_turns`.
- `success_evaluator()` → score the final boxed answer with `recipe/sod/reward.py`
  (math_dapo strict-box) → avg@32.
- register in `agent_system/environments/env_manager.py::make_envs` + a `*_projection`.
Then run via `verl.trainer.main_ppo` with `env.env_name=math_tool`, `val_only=True`,
`val_kwargs.n=32 temperature=1.0 top_p=0.6 top_k=20`, `multi_turn.enable=True`,
`tool_config_path=…/sandbox_fusion_tool_config.yaml`.

## Faithfulness caveats (vs SOD)
- SOD used **vllm + hermes** tool format; SDAR uses **sglang + chatml** (unavoidable here).
- SOD wraps the sandbox in a `CustomSandboxFusionTool` (strips ```python fences, auto-prints
  the last line); SDAR's stock `SandboxFusionTool` does neither — tool outputs may differ slightly.
- SOD adds a small turn-count penalty to *negative* scores; we report **avg@32 accuracy**
  (unaffected by that shaping), the headline SOD metric.
- top_k=20 is set explicitly (SOD's model `generation_config` default; closes most of the
  faithful-vs-paper gap per the SOD reproduction).

## Reproduce the data
```
python recipe/sod/data_preprocess_sod.py \
    --src_dir <SOD>/Open-AgentRL-Eval --out_dir ~/data/sod_aime   # -> test.parquet (60)
python recipe/sod/test_sod_scorer.py   # reward unit test (sdar env)
```
