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

## SOD-1.7B full run (2026-07-02) — TWO problems found (honest)

Sharded run (6 shards × 10 problems, faithful top_k=20/20480/16). Only 2/6 shards produced
scores; **4/6 HUNG for 12–14 h** (no log output; `multi_turn_loop` appears to deadlock/stall on
shards with many degenerate-loop rollouts) — a reliability bug in the env-step rollout under
SOD-1.7B's looping. The 2 that finished:

| shard | AIME2024 | AIME2025 | tool_call_count/mean |
|---|---|---|---|
| shard5 | 14.4% | 6.25% | ~0.8 |
| shard2 | (n/a) | 0.0% | ~0.9 |

**KEY FINDING — SOD-1.7B underperforms badly in the SDAR harness (~14/6 vs standalone 48/37).**
Cause = **tool-format mismatch**: SOD-1.7B was trained for the **hermes** tool protocol (vllm,
tool schema injected via `apply_chat_template(tools=...)`). SDAR's `math_tool` env presents the
problem as plain **chatml** text and parses `<tool_call>` from the raw output *without* injecting
the tool schema the model expects → the model rarely emits a valid tool call
(**tool_call_count ~0.8** here vs ~5 for SOD's working samples) → far lower accuracy.

**Conclusion:** the `math_tool` benchmark is built + smoke-validated (the pipeline runs end-to-end),
but it is **NOT a faithful reproduction of SOD's number** for SOD-1.7B, and it is currently
**unreliable** (hangs on degenerate-heavy shards). A faithful SOD number needs (a) injecting the
tool schema in hermes format the model was trained on, and (b) fixing the multi_turn_loop hang.
Until then, the **standalone SOD reproduction (48.12/36.77, repo explcre/sod-repro) is the
faithful number**; the SDAR `math_tool` result stands as "benchmark works, SOD-1.7B scores ~14/6
under the chatml/env tool protocol."

## Reference — standalone SOD reproduction (repo explcre/sod-repro, for comparison)
| config | AIME2024 | AIME2025 |
|---|---|---|
| faithful (top_k off) | 42.08 | 30.52 |
| **top_k=20** | **48.12** | **36.77** |
| paper | 50.83 | 41.72 |
The SDAR `math_tool` run targets the same top_k=20 protocol; expect it to land in the same
ballpark once it completes (the sglang→env-step tool mechanism differs from SOD's vllm+hermes,
so some drift is expected — documented in README "Faithfulness caveats").
