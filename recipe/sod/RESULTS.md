# SOD benchmark in SDAR — results log

The `math_tool` AIME tool-integrated-reasoning benchmark added to SDAR (see README.md).

## Pipeline validation (GPU + SandboxFusion) — ✅ PASSED
Smoke run (Qwen2.5-3B-Instruct, 4 problems, avg@2), voyager/laniakea:
- `val/aime2024_success_rate = 0.125`, `val/aime2025_success_rate = 0.25`
- `tool_call_count > 0` on aime2025 → the `code_interpreter` tool fires, SandboxFusion executes
  the Python, `<tool_response>` returned, boxed answer scored by `recipe/sod/reward` (math_dapo
  strict box), per-source `success_rate` reported.
=> the whole env → code-tool → sandbox → strict-box → avg@N pipeline runs end-to-end.

## SOD-1.7B full avg@32 (60 problems) — IN PROGRESS / expensive
- Faithful sampling temp 1.0 / top_p 0.6 / **top_k 20**, max_resp 20480, 16 turns, avg@32.
- Fixes needed to run: (1) `micro_batch_size_per_gpu` + `enable_chunked_prefill` +
  `max_num_batched_tokens` (verl config validation); (2) `max_prompt_length` 4096 -> **16384**
  and `truncation=left` — the env re-prompts with the growing multi-turn transcript, which
  overflowed 4096 (`NotImplementedError: seq 6720 > 4096`).
- STATUS (2026-07-02): running on voyager (H100), ~7 h in, **grinding the degenerate-loop
  rollouts** — SOD-1.7B's known ~45% "announce code but never emit `<tool_call>`" behavior
  (GPU busy generating, sandbox idle) fills 16 turns × 20480 tokens per looping rollout, so the
  tail is very slow; voyager preemption repeatedly requeued it. No aggregate reported yet.
- To get a number faster: bound the degenerate generations (`MAX_RESP=8192 MAX_TURNS=8`) and/or
  fewer problems — trades a little faithfulness for tractable wall-clock.

## Reference — standalone SOD reproduction (repo explcre/sod-repro, for comparison)
| config | AIME2024 | AIME2025 |
|---|---|---|
| faithful (top_k off) | 42.08 | 30.52 |
| **top_k=20** | **48.12** | **36.77** |
| paper | 50.83 | 41.72 |
The SDAR `math_tool` run targets the same top_k=20 protocol; expect it to land in the same
ballpark once it completes (the sglang→env-step tool mechanism differs from SOD's vllm+hermes,
so some drift is expected — documented in README "Faithfulness caveats").
