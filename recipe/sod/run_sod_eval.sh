#!/bin/bash
# SOD benchmark (AIME-TIR) evaluation on SDAR via its NATIVE agentic rollout.
#
# Path: verl.trainer.main_ppo (val_only) -> agent_system.make_envs -> the new
# `math_tool` environment. The model emits a hermes code_interpreter tool call,
# the env runs the Python in SandboxFusion and returns <tool_response>...,
# then the model boxes its final answer which is scored by recipe/sod/reward
# (math_dapo strict box, the SAME scorer SOD uses). avg@N with SOD's sampling.
#
# Prereqs:
#   1) Build data (NOW emits an env_kwargs column the agent loop needs):
#        python recipe/sod/data_preprocess_sod.py --src_dir <Open-AgentRL-Eval> --out_dir <DATA_DIR>
#   2) A SandboxFusion server reachable at $SANDBOX_URL (default localhost:8080/run_code).
#      If $SANDBOX_SIF is set, this script starts it via singularity (as in the SOD repo).
#
# Env knobs: MODEL_PATH, DATA_DIR, SANDBOX_URL, SANDBOX_SIF, NGPUS, INFER_TP,
#            VAL_N (avg@N), VAL_TEMP, VAL_TOP_P, VAL_TOP_K, MAX_PROMPT, MAX_RESP,
#            MAX_TURNS, HIST_LEN, N_PROBLEMS.
set -x
PROJECT_DIR="$(pwd)"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH (e.g. the SOD-1.7B ckpt or a Qwen3/Qwen2.5 model)}
DATA_DIR=${DATA_DIR:?set DATA_DIR (dir with test.parquet from data_preprocess_sod.py)}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8080/run_code}
NGPUS=${NGPUS:-1}; INFER_TP=${INFER_TP:-1}
VAL_N=${VAL_N:-32}; VAL_TEMP=${VAL_TEMP:-1.0}; VAL_TOP_P=${VAL_TOP_P:-0.6}; VAL_TOP_K=${VAL_TOP_K:-20}
MAX_PROMPT=${MAX_PROMPT:-4096}; MAX_RESP=${MAX_RESP:-20480}; MAX_TURNS=${MAX_TURNS:-16}
HIST_LEN=${HIST_LEN:-8}
# AIME2024(30) + AIME2025(30) = 60 problems by default; val_envs must cover N_PROBLEMS*VAL_N.
N_PROBLEMS=${N_PROBLEMS:-60}
VAL_BS=$(( N_PROBLEMS * VAL_N ))

# optional: start SandboxFusion via singularity (same image SOD uses); else assume $SANDBOX_URL is up
SBX_PID=""
if [ -n "${SANDBOX_SIF:-}" ]; then
  SING=$(command -v singularity || echo /pkg/singularity/3.8.3/bin/singularity)
  nohup "$SING" run --bind /tmp:/tmp "$SANDBOX_SIF" > sandbox_sod.log 2>&1 &
  SBX_PID=$!; trap 'kill $SBX_PID 2>/dev/null' EXIT
  for i in $(seq 1 120); do curl -sf "${SANDBOX_URL%/run_code}/v1/ping" >/dev/null 2>&1 && break; sleep 5; done
fi
# fail fast if the sandbox/tool endpoint is down (a TIR eval without a sandbox is invalid)
python3 -c "import requests,sys; r=requests.post('$SANDBOX_URL',json={'code':'print(6*7)','language':'python'},timeout=30); sys.exit(0 if r.status_code==200 and '42' in r.text else 1)" \
  || { echo "FATAL: sandbox not reachable at $SANDBOX_URL"; exit 1; }

export PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_files="$DATA_DIR/test.parquet" \
    data.val_batch_size=$VAL_BS \
    data.train_batch_size=$N_PROBLEMS \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    data.return_raw_chat=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ:-8} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ:-8} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ:-8} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED:-32768} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMP \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    env.env_name=math_tool \
    env.max_steps=$MAX_TURNS \
    env.history_length=$HIST_LEN \
    env.rollout.n=1 \
    env.math_tool.sandbox_fusion_url="$SANDBOX_URL" \
    reward_model.reward_manager=episode \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    "$@"
echo "=== SOD BENCHMARK EVAL DONE (avg@${VAL_N}; success_rate per data_source: aime2024 & aime2025) ==="
