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

**The `math_tool` agent environment (the native SDAR integration — implemented + CPU-tested):**
Architecture finding: SDAR routes **all** multi-turn rollout — *including tool use* — through an
**`agent_system` environment** (`_validate()` → `traj_collector.multi_turn_loop(envs=…)`, which
*requires* an `EnvironmentManagerBase`); there's no stock-verl pure-tool path, and `main_ppo`
only accepts `reward_manager=episode`. So the SOD AIME-TIR eval is added as a proper env:
- `agent_system/environments/env_package/math_tool/{envs.py,projection.py,__init__.py}` — a
  per-sample `MathToolEnv` (+ threaded vectorized backend, mirroring the `search` env): `reset()`
  presents SOD's exact AIME prompt (from `env_kwargs.question`); `step()` parses a hermes
  `<tool_call>{...code_interpreter...}` (tolerant `json.loads(strict=False)`), runs the Python via
  verl's `_process_single_case` (the SandboxFusion helper) and returns `<tool_response>…`; a final
  `\boxed{}` is scored by `recipe/sod/reward.py` (math_dapo strict-box) → `won`.
- `MathToolEnvironmentManager` + `MathToolMemory` in `env_manager.py`/`memory.py` (multi-turn
  transcript), a `make_envs` branch for `env.env_name=math_tool`, and an `env.math_tool` config
  block in `ppo_trainer.yaml`. `success_evaluator` reports avg@N accuracy per `data_source`.
- CPU test: `test_math_tool_env.py` (reset/boxed-scoring/tool-call/projection/vectorized — passes,
  sandbox mocked). Reward path uses `reward_manager=episode` (env scores internally).

**Run (GPU + a SandboxFusion server):**
```
python recipe/sod/data_preprocess_sod.py --src_dir <Open-AgentRL-Eval> --out_dir <DATA_DIR>
MODEL_PATH=<ckpt> DATA_DIR=<DATA_DIR> SANDBOX_URL=http://localhost:8080/run_code \
  bash recipe/sod/run_sod_eval.sh   # val_only avg@32, env.env_name=math_tool, top_k=20
```
avg@32 lands in `val-core/.../aime2024 & aime2025 success_rate`. (Pending: a GPU+sandbox smoke run.)

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
