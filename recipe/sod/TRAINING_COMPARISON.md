# SOD training vs SDAR — how the training step differs, time & hardware

**Scope note (important):** what we added to SDAR is the **evaluation** benchmark only
(`math_tool` env + AIME TIR eval, `val_only=True`). **No training was run, and SOD-style
training is NOT implemented in SDAR.** This doc answers "if we trained, how would it differ
from the SOD paper, and what would it cost." All SOD numbers below are read directly from
`~/text-dna/SOD/examples/SOD/run_sod.sh` and the SOD README (arXiv 2605.07725).

---

## 1. Is the training step different from the SOD paper? — YES, fundamentally.

| | **SOD paper** (`run_sod.sh`) | **SDAR paper** (GiGPO) | **What we built** |
|---|---|---|---|
| Paradigm | GRPO **+ step-wise on-policy distillation (OPD)** | GiGPO self-distilled RL | *eval only* |
| Teacher | **YES** — GRPO-trained Qwen3-4B loaded as `ref.model` | none (teacher-free) | n/a |
| Core loss | policy-grad (GRPO adv) **+ per-step-weighted token-KL to teacher** | policy-grad (group-in-group adv), **no distillation term** | n/a |
| SOD's actual novelty | `token_kl_reg.stepwise_*`: **adaptively down-weights the OPD loss on steps where the student drifted from the teacher** (suppresses cascading TIR error) | — | not ported |
| Credit assignment | step-wise (per reasoning step) | group-in-group (turn + trajectory) | n/a |
| Rollout mechanism | native vllm, one continuous assistant sequence, inline tool pause/resume | env-step: re-templates full transcript as a new user turn each step | env-step (eval) |

**Bottom line:** SOD's entire contribution is the *step-wise weighted OPD term against a 4B
teacher*. SDAR's trainer is teacher-free GiGPO and has **no such term**. So "training in SDAR"
is NOT "training the SOD way" — it would be a *different method*. To reproduce SOD's number by
training, you use SOD's own repo (`run_sod.sh` is ready); to train SOD-style *inside* SDAR you'd
have to port the `token_kl_reg` step-wise module into SDAR's trainer (a real implementation task,
not done). Note also the env-step rollout under-tool-calls (~1 vs ~5, see RESULTS.md) — that same
gap would depress SDAR-side *training* rollouts, not just eval.

---

## 2. SOD paper training recipe (exact, from run_sod.sh)

- Algorithm: `adv_estimator=grpo`, GRPO group `rollout.n = 16`
- Teacher: `ref.model.path = SOD-GRPO_teacher-4B` (downloadable from HF — Step-1 teacher training skippable)
- OPD: `token_kl_reg.enable=True`, `stepwise_enable=True`, `stepwise_delta=0.2`, `stepwise_opd_coef=1.0`, `teacher_kl_coef=0.002`
- Data: **Open-AgentRL-30K** (30K), `train_batch_size=64`, `ppo_mini_batch_size=16`, **1 epoch** → ~469 optimizer steps
- Lengths: `max_prompt=2560`, `max_response=20480`; `actor_lr=1e-6`
- Per step: 64 prompts × 16 rollouts = **1024 multi-turn TIR rollouts** (≤20480 tok + SandboxFusion code exec) + teacher-4B forward for the OPD KL + actor update
- **Hardware (paper): 8× NVIDIA H20 96GB, batch size 64, 1 node**

---

## 3. Time & hardware estimates

Wall-clock is dominated by the 1024 long TIR rollouts/step (vllm gen + serialized sandbox code
exec) plus the extra teacher-4B forward. The SOD paper does not publish a wall-clock; these are
engineering estimates for the 30K / 1-epoch / 469-step recipe.

| Hardware | Fits? | Est. wall-clock (30K, 1 epoch) | Notes |
|---|---|---|---|
| 8× H20 96GB (paper) | ✅ | **~2–3 days** | reference config |
| **voyager 4× H100 80GB** | ✅ (tight mem) | **~2–4 days** | fastest per-GPU; half the GPUs, 320GB agg. Fair-share cap ≤3 running jobs; 1 job = up to 4 GPUs on the node. **Recommended for us.** |
| laniakea 8× RTX6000-Ada 49GB | ✅ (tight) | ~4–6 days | most GPUs but slower each; 4B teacher + 1.7B + 20480 KV crowds 49GB → needs tp + vllm offload |
| galaxy 6× 3090 24GB | ⚠️ not advised | — | 24GB too small for 4B-teacher + 20480 KV comfortably |

Estimate basis: ~3–8 min/optimizer step (rollout-bound; sandbox exec is the serial bottleneck) ×
469 steps. The sandbox throughput (parallel SandboxFusion workers) matters as much as GPU FLOPs —
under-provisioning it can double the wall-clock regardless of hardware.

**Cheaper knobs** if a faithful 2–3 day run is too much: SOD-0.6B student instead of 1.7B; shorter
`max_response` (12288); a 10–15K data subset; fewer GRPO rollouts (n=8). Each trades some fidelity
to the paper number for wall-clock.

---

## 4. Recommendation

- **To reproduce SOD's number by training:** run it in **SOD's own repo** (`examples/SOD/run_sod.sh`,
  teacher = downloadable `SOD-GRPO_teacher-4B`) on **voyager 4× H100**, budget **~2–4 days**. This is
  the faithful, paper-matching path — SDAR's trainer would give a *different* (non-SOD) method.
- **To train SOD-style inside SDAR:** first port the `token_kl_reg` step-wise OPD module + teacher-ref
  into SDAR's GiGPO trainer (not done). Only worth it if the goal is specifically "SOD-method under
  SDAR's env-step agentic rollout."
- **For just a number now:** the standalone SOD *eval* reproduction (48.1/36.8, explcre/sod-repro) and
  the SDAR `math_tool` *eval* (24.6/23.5) are both done — no training needed for either.
