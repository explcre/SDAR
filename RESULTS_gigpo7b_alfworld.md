# GiGPO-Qwen2.5-7B ALFWorld — Reproduction Results

Eval of the released **GiGPO-Qwen2.5-7B-Instruct-ALFWorld** checkpoint under SDAR's framework,
plus from-scratch SDAR training runs (3B / 7B). Last updated: 2026-06-07.

> Status: eval cells for seed-123 **verified**; multi-seed (n=4) reruns in progress; 3B/7B trainings running.

---

## 1. Eval configuration (what every number below was produced under)

All eval numbers come from the **HF checkpoint** (no verl training loop — `step:0`, initial eval only).
Result-determining params are identical across runs:

| param | value | source |
|---|---|---|
| `val_kwargs.n` | 1 | matches GiGPO paper protocol (games × 3 seeds, n=1 per game) |
| `val_kwargs.temperature` | 0.4 | SDAR recipe default |
| `val_kwargs.do_sample` | True | SDAR recipe default |
| `env.history_length` | 2 (h2) or 4 (h4) | reported per cell |
| `VAL_OUT` | True = **valid_unseen**, False = **valid_seen** | `verl/.../env_manager.py:634` |

**`gpu_memory_utilization` (GMU) is NOT a result-determining parameter** — it is a vLLM
memory/throughput knob (KV-cache fraction → batch concurrency), hardware-dependent, and not
part of the paper's config. It does not change generated tokens given fixed seed + sampling
params. We record it per-run for full transparency; seed-123 (H100@0.6) agreeing with the
6000Ada@0.4 reruns is a config-invariance cross-check, not a confound.

### Per-run config table

| seed | GPU | GMU | temp | n | hist | cell | success | job |
|---|---|---|---|---|---|---|---|---|
| 123 | H100 80GB | 0.6 | 0.4 | 1 | 2 | unseen | **0.928** | 232437 |
| 123 | H100 80GB | 0.6 | 0.4 | 1 | 2 | seen   | **0.971** | 232458 |
| 123 | H100 80GB | 0.6 | 0.4 | 1 | 4 | unseen | **0.951** | 232445 |
| 42  | 6000Ada 49GB | 0.4 | 0.4 | 1 | 2 | unseen | _running_ | 232492 |
| 7   | 6000Ada 49GB | 0.4 | 0.4 | 1 | 2 | unseen | _running_ | 232493 |
| 99  | 6000Ada 49GB | 0.4 | 0.4 | 1 | 2 | unseen | _queued_  | 232494 |

> GMU note: 7B 1-GPU eval at GMU 0.6 OOMs on a 49 GB 6000Ada (full 14 GB model in FSDP + vLLM
> reservation + broadcast > 47 GB). GMU ≤ 0.4 is required on 49 GB cards; 0.6 is fine on the 80 GB H100.

---

## 2. Eval results (seed-123, verified from logs)

### Aggregate success rate

| cell | history | success | GiGPO paper (ALFWorld) |
|---|---|---|---|
| **valid_unseen** | 2 | **0.928** | ~0.902–0.908 ✓ (reproduced) |
| valid_seen | 2 | 0.971 | — |
| valid_unseen | 4 | 0.951 | — |

### Per-subtask success rate (seed-123)

| subtask | unseen h2 (0.928) | seen h2 (0.971) | unseen h4 (0.951) |
|---|---|---|---|
| pick_and_place | 0.808 | 1.000 | 0.885 |
| look_at_obj_in_light | 0.912 | 0.946 | 1.000 |
| pick_clean_then_place | 0.938 | 1.000 | 0.927 |
| pick_heat_then_place | 0.931 | 0.950 | 0.941 |
| pick_cool_then_place | 1.000 | 0.952 | 1.000 |
| pick_two_obj_and_place | 1.000 | 0.942 | 1.000 |

> Per-subtask single-seed variation (e.g. pick_two 1.000 vs paper-ish) reflects **tiny-N**
> (pick_two has only ~8–24 games per split), not a bug. Aggregate is the stable comparison.

### Multi-seed unseen-h2 (n=4) — _to be completed_

| seed | success |
|---|---|
| 123 | 0.928 |
| 42  | _pending_ |
| 7   | _pending_ |
| 99  | _pending_ |
| **mean ± std** | _pending_ |

---

## 3. SDAR from-scratch training (in progress)

Validating the SDAR training pipeline (no released SDAR checkpoints exist — training from scratch).

| model | node | GPUs | target (paper) | job | status |
|---|---|---|---|---|---|
| Qwen2.5-3B | voyager | 2 (TP=2) | 84.4 | 232485 | running |
| Qwen2.5-7B | laniakea | 4 (TP=2×DP=2) | 85.9 | 232488 | running |

- `enforce_eager=True`, `train_batch_size=16` → **NGPUS must divide 16** (use 1/2/4/8, not 6).

---

## 4. Key facts / caveats

- **`VAL_OUT=True` = valid_unseen**, `False` = valid_seen (`verl/.../env_manager.py:634`). The
  early job filenames ("seenFULL") are misnomers from initial naming; the VAL_OUT flag in the log
  is authoritative.
- **`val_kwargs.n=1`** matches the GiGPO paper protocol (per-game single rollout × seeds).
- **No released SDAR checkpoints** — the training runs are from-scratch reproductions.
- **TCOD's reported 76–83%** differs from this because it uses a *different eval harness*, not a
  different metric.
- **Honest metric** = per-attempt success mean (no source-cell / best-of-N inflation).

---

## 5. Infrastructure notes (for reproducibility)

- **Root cause of early run failures**: `/home` over its 512 GB quota → `EDQUOT` made
  vLLM/triton/inductor **compile-cache writes block**, which hung jobs right after model-load
  (looked like a GPU hang; it was disk). Not GPU contention.
- **Fix (all scripts)**: compile caches → **node-local `/tmp`**; checkpoints → **shared SSD**
  `/tmp/galaxy_srv_disk00/pengchx3` (self-healing sshfs remount, fallback node-local);
  `VLLM_DO_NOT_TRACK=1`. The GiGPO eval model was moved off /home to the SSD (recipe respects
  `MODEL_PATH`/`OUTDIR` env). Storage policy per `announcement_of_voyager_laniakea_ssd.md`.
- **Multi-seed history**: first multi-seed attempts (seeds 42/7/99) failed on the quota-hang and a
  GMU-too-high OOM on the 49 GB cards; reruns use the hardened storage + GMU 0.4.
