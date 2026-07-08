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

## ROOT-CAUSE DIAGNOSIS of the eval gap (2026-07-04) — flat-user vs native hermes multi-turn

Q: eval is on the OPEN-SOURCED youngzhong/SOD-1.7B (same weights both harnesses) — why does SDAR
get 24.6/23.5 vs standalone 48/37? A: NOT the ckpt. The SDAR env re-serializes the whole
conversation into ONE narrated `role:user` message every turn, instead of the native hermes
multi-turn (role:assistant for the model's own turns, role:tool for tool results) SOD-1.7B was
trained on.

Evidence in code:
- env_manager.py MathToolEnvironmentManager.build_text_obs (L181-209) formats each turn as
  MATH_TOOL_TEMPLATE(task, memory_context, step_count) -> a SINGLE string.
- math_tool.py MATH_TOOL_TEMPLATE (L18-25): "{problem} Prior to this step you have already taken
  N step(s). Below is the interaction history ... {memory_context} Now continue ..." — the whole
  transcript as PROSE inside one user turn.
- rollout_loop.py preprocess_single_sample (L106-124): chat = [{"role":"user","content":obs_text}]
  — always one user message, add_generation_prompt=True, re-templated fresh each step.
- rollout_loop.py L398: skip_special_tokens=True strips structural tokens before the action is stored.

Four OOD breaks vs SOD-1.7B training: (1) model's own prior turns quoted back as USER text, never
role:assistant -> it can't continue its own reasoning chain; (2) prose narration not in training
distribution; (3) tool results not role:tool/inline <tool_response>; (4) special tokens stripped.
=> model won't chain tool calls (tool_call ~1 vs ~5) => ~half the score. Matches the exact 24.6 vs 48.

FIX (in progress): accumulate a real hermes message list (user problem -> assistant raw-gen ->
tool response -> assistant ...) and apply_chat_template on the FULL list; flag-gated so other
SDAR envs are unaffected. Expected to recover toward 48/37 if the diagnosis is correct.

## NATIVE-MULTITURN FIX — smoke CONFIRMS the diagnosis (2026-07-04)

shard0 (same 10 problems), avg@8, bounded 8192/8, SOD-1.7B. Old flat-narration vs native hermes:

| metric | old (flat user msg) | NATIVE hermes multi-turn |
|---|---|---|
| tool_call_count/mean | ~1 | **2.3** (climbing toward SOD-native ~5) |
| aime2024 | 13.8 | **37.5** |
| aime2025 | 55.6 | **87.5** |
| overall | ~34.7 | **62.5** |

Diagnosis CONFIRMED: the flat-user re-serialization was the cause. Native hermes (role:assistant
for the model's own turns + role:tool results) roughly doubles the score and lifts tool-calling.
Two integration bugs fixed en route (N_PROBLEMS must equal data problem-count; chat must be
np.array(dtype=object) for downstream .tolist()). Launching full faithful (20480/16) avg@32.

## FULL FAITHFUL NATIVE RUN (2026-07-07) — native fix recovers most of the gap

Sharded avg@32, faithful 20480/16, all 60 problems, native hermes multi-turn ON. Per-shard
(clean laniakea sandboxes except galaxy shard2 which suffered SandboxFusion overload):

| shard | node | aime2024 | aime2025 | note |
|---|---|---|---|---|
| 0 | laniakea | 42.5 | 83.8 | |
| 1 | laniakea | 53.7 | 29.4 | |
| 2 | galaxy | 41.2 | 0.0 | CORRUPT: 2566 sandbox "handler closed" errors -> aime2025 collapsed; re-running clean |
| 3 | laniakea | 22.5 | 53.1 | |
| 4 | laniakea | 23.1 | 18.8 | (clean rerun; galaxy orig discarded) |
| 5 | laniakea | 18.8 | 18.8 | |

Provisional aggregate (incl. corrupt galaxy shard2): **aime2024 = 33.6, aime2025 = 34.0**.
Clean shard2 rerun (sdar-s2clean) pending -> will lift aime2025 (siblings are 18-84, not 0).

| config | AIME2024 | AIME2025 |
|---|---|---|
| SDAR OLD flat-narration | 24.6 | 23.5 |
| **SDAR native hermes (faithful, provisional 6/6)** | **33.6** | **34.0** |
| standalone SOD top_k=20 | 48.1 | 36.8 |
| paper | 50.8 | 41.7 |

TAKEAWAY: the native-hermes multi-turn fix (role:assistant/role:tool preserved instead of flat
narration) lifts SDAR from 24.6/23.5 to ~33.6/34.0 — aime2025 now MATCHES standalone (34.0 vs
36.8), aime2024 substantially closes (33.6 vs 48.1). Confirms the diagnosis: the eval gap was the
harness prompt format, not the checkpoint. Residual aime2024 gap + variance is partly SandboxFusion
reliability under heavy native tool-calling (an infra bottleneck, not the fix). Clean smoke on
shard0 (avg@8) hit 62.5 overall, further corroborating.

## COMPUTE COST (measured from job logs, 2026-07-07) — EVAL ONLY, no training run

Everything run in this effort was EVAL (val_only=True). No SOD/SDAR TRAINING was run; training
numbers below are config estimates (see TRAINING_COMPARISON.md).

MEASURED eval wall-clock:
| eval | harness | hardware | time |
|---|---|---|---|
| standalone SOD, full AIME avg@32 (60 prob) | vLLM (tool loop INSIDE one continuous generation, KV reused) | 1x galaxy 3090 | ~1-2 h |
| SDAR smoke (bounded 8192/8, 4-10 prob) | env-step | 1 GPU | ~2-30 min |
| SDAR faithful 20480/16 avg@32, 10 prob/shard | env-step | laniakea 6000-Ada | ~11 h/shard (11:10, 10:49) |
| " | env-step | voyager H100 | ~3-4 h/shard |
| " | env-step | galaxy 3090 | pathological 1d+ (sandbox overload) |
| SDAR full 60 prob (6 shards parallel) | env-step | 6x laniakea | ~11 h wall-clock |

KEY: BOTH harnesses use vLLM. SDAR is ~5-10x SLOWER not because of the engine but the ORCHESTRATION:
SOD runs the tool loop INSIDE one continuous vLLM generation (KV cache reused, all rollouts batched,
tool result spliced inline). SDAR calls vLLM generate_sequences ONCE PER TURN from a Python env-step
loop -> (1) KV cache NOT reused (re-prefills the growing prompt each turn), (2) per-turn barrier (batch
waits for slowest), (3) serial sandbox round-trips. Same engine, different multi-turn wrapper.

TRAINING (NOT run; estimates): SOD run_sod.sh (GRPO+step-wise OPD, Qwen3-4B teacher, 30K, 1 epoch
~469 steps) = 8x H20 96GB ~2-3 d (paper); voyager 4x H100 ~2-4 d; laniakea 6x 6000-Ada ~5-8 d.

## FINAL CLEAN 6/6 (2026-07-07) — native hermes, faithful 20480/16 avg@32

Galaxy-corrupted shard2 re-run on a clean laniakea sandbox (galaxy 41.2/0.0 -> CLEAN 68.1/1.9:
the sandbox overload had HALVED aime2024; aime2025~2 is genuine — those 5 problems are just hard).

Per-shard (clean): a24=[42.5,53.7,68.1,22.5,23.1,18.8] a25=[83.8,29.4,1.9,53.1,18.8,18.8]

| config | AIME2024 | AIME2025 |
|---|---|---|
| SDAR OLD flat-narration | 24.6 | 23.5 |
| **SDAR NATIVE hermes (clean 6/6)** | **38.1** | **34.3** |
| standalone SOD top_k=20 | 48.1 | 36.8 |
| paper | 50.8 | 41.7 |

HEADLINE: the native-hermes multi-turn fix lifts SDAR from 24.6/23.5 -> 38.1/34.3. aime2025 reaches
~93% of standalone (34.3 vs 36.8); aime2024 closes ~55% of the old gap (24.6->38.1 vs 48.1). Confirms
the diagnosis definitively: the eval gap was the harness prompt format (flat-narration vs native
role:assistant/role:tool), NOT the checkpoint. Residual gap to standalone is (a) env-step per-turn
re-templating still isn't a single continuous vLLM sequence, (b) SandboxFusion reliability under load,
(c) benchmark variance (30 problems/source; shard2's aime2025 problems are outlier-hard).

Also fixed aggregate_sharded.py: prefer the final compact summary line (the np.float64 metric-dict
print can omit one source, which silently undercounted the mean).
