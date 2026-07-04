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

## ROOT CAUSE + FIX (2026-07-02) — it was `enable_thinking`, NOT tool-format

CORRECTION: the "tool-format mismatch" conclusion below was WRONG. Real root cause:
SOD-1.7B (Qwen3) was fine-tuned with **thinking mode OFF** (chat_template.jinja appends an empty
`<think>\n\n</think>\n\n` block when `enable_thinking=False`, then plain reasoning). The
standalone SOD eval passed `enable_thinking=False`; SDAR's `apply_chat_template` defaulted it ON
-> the model was out-of-distribution -> emitted `<|im_start|>` control-token garbage -> ~14/6 AND
the multi_turn_loop "hangs" (garbage filled the 20480-token budget). The Qwen2.5-3B smoke didn't
catch it (Qwen2.5 has no thinking mode). rope/config was ruled out (transformers 4.57.3 reads
rope_theta=1000000 correctly). tool-schema was ruled out (standalone had none either).

FIX (config-level, no code change): `+data.apply_chat_template_kwargs.enable_thinking=False`.
CONFIRMED (smoke, SOD-1.7B avg@2, 4 problems): responses coherent ("I'll solve this step by
step."), val/aime2024=0.5, val/aime2025=1.0, 0 crashes. Full faithful sharded run relaunched.

## SOD-1.7B full run (2026-07-02) — TWO problems found (SUPERSEDED by the fix above)

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

## FINAL ASSESSMENT (2026-07-03) — fix works, but SDAR harness gives SOD-1.7B low scores

After the enable_thinking fix, responses are genuinely COHERENT ("I'll solve this step by step
using the power of ...") — the fix is real. But the aggregate SOD-1.7B avg@32 in SDAR is LOW,
in BOTH configs and NOT a small-sample fluke:
- faithful (20480/16): shard5 aime2025 = 0.10
- bounded  (8192/8):   shard2 aime2025 = 0.00, shard5 aime2024 = 0.075
- tool_call_count/mean ~0.7-1.15  (vs SOD's native ~5 tool calls per solved sample)

So SOD-1.7B scores ~7-10% in SDAR's env-step harness vs 48/37 standalone — it's a HARDER harness
for this model. Root causes (beyond the fixed enable_thinking):
1. env-step re-templates the FULL transcript as a single user turn each step; SOD's native
   vllm+hermes keeps one growing assistant sequence with the tool schema in the system prompt.
   The model tool-calls far less here (~1 vs ~5) -> solves far fewer.
2. reliability at avg@32 scale: 2/6 shards FAILED on transient Ray-GCS init on contended nodes,
   2/6 CANCELLED at start; faithful config takes ~7 h/shard (genuine SOD loops + env overhead).

CONCLUSION: the `math_tool` benchmark is a valid TIR eval and the pipeline + fix are validated,
but it does NOT reproduce SOD's paper number for SOD-1.7B (the env-step tool mechanism differs
materially from SOD's native vllm+hermes). A faithful SOD number would require porting SOD's
native tool-rollout (hermes, schema-injected, single-sequence) — a large deviation from SDAR's
agentic design. The **standalone SOD reproduction (48.12/36.77, explcre/sod-repro) remains the
faithful number**; the SDAR result is "benchmark works, SOD-1.7B scores ~7-10% under env-step TIR."

## RESULT with BOTH fixes (2026-07-04) — 24.6/23.5, up 3x from broken, still ~half standalone

Both fixes applied: `enable_thinking=False` + code_interpreter tool-SCHEMA injection (hermes
`<tools>` system block). Sharded avg@32, all 60 problems, BOUNDED (MAX_RESP=8192/MAX_TURNS=8):

| | AIME2024 | AIME2025 |
|---|---|---|
| SDAR broken (before fixes) | ~7 | ~10 |
| **SDAR both fixes (bounded avg@32)** | **24.60** | **23.54** |
| standalone SOD top_k=20 | 48.12 | 36.77 |
| paper | 50.83 | 41.72 |

Per-shard (5 problems each, high variance): 2024=[13.8,41.8,60.0,4.8,21.2,6.0], 2025=[55.6,26.9,0,33.8,15,10].

So the two config fixes are REAL and large (7->24.6, ~3x). Residual gap to standalone (48/37):
1. **tool_call_count ~1 vs SOD-native ~5** — even with the schema injected, the env-step loop
   (re-prompt whole transcript each turn) yields far fewer tool calls than SOD's single-sequence
   vllm+hermes -> less arithmetic verification -> ~half the accuracy.
2. **bounded MAX_RESP=8192/MAX_TURNS=8** vs SOD's faithful 20480/16 — cuts long reasoning; a
   faithful run (now that gen is coherent, the earlier hangs should be gone) may recover some,
   but ~7 h/shard.

VERDICT: the SDAR `math_tool` benchmark now evaluates SOD-1.7B sanely (24.6/23.5), the two root
causes (enable_thinking, tool-schema) are found+fixed, but the env-step tool mechanism inherently
gives ~half of SOD's native harness. Standalone 48/37 remains the faithful paper-comparison number.
