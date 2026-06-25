#!/bin/bash
# SOD benchmark (AIME-TIR) evaluation on SDAR/verl.
# Reuses SDAR infra: verl.trainer.main_ppo (val_only), sglang_async multi-turn rollout,
# the code_interpreter tool (verl.tools.sandbox_fusion_tools), and the math_dapo scorer
# (routed via data_source=aime2024/aime2025). avg@32 with SOD's sampling.
#
# Prereqs:
#   1) Build data:  python recipe/sod/data_preprocess_sod.py --src_dir <Open-AgentRL-Eval> --out_dir <DATA_DIR>
#   2) A SandboxFusion server reachable at $SANDBOX_URL (default localhost:8080/run_code).
#      If $SANDBOX_SIF is set, this script starts it via singularity (as in the SOD repo).
# Env knobs: MODEL_PATH, DATA_DIR, SANDBOX_URL, SANDBOX_SIF, NGPUS, INFER_TP,
#            VAL_N (avg@N), VAL_TEMP, VAL_TOP_P, VAL_TOP_K, MAX_PROMPT, MAX_RESP, MAX_TURNS.
set -x
PROJECT_DIR="$(pwd)"
TOOL_CFG="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH (e.g. the SOD-1.7B ckpt or a Qwen3/Qwen2.5 model)}
DATA_DIR=${DATA_DIR:?set DATA_DIR (dir with test.parquet from data_preprocess_sod.py)}
SANDBOX_URL=${SANDBOX_URL:-http://localhost:8080/run_code}
NGPUS=${NGPUS:-1}; INFER_TP=${INFER_TP:-1}
VAL_N=${VAL_N:-32}; VAL_TEMP=${VAL_TEMP:-1.0}; VAL_TOP_P=${VAL_TOP_P:-0.6}; VAL_TOP_K=${VAL_TOP_K:-20}
MAX_PROMPT=${MAX_PROMPT:-4096}; MAX_RESP=${MAX_RESP:-20480}; MAX_TURNS=${MAX_TURNS:-16}

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

# write the tool config's sandbox url to match $SANDBOX_URL at runtime (keeps the yaml env-agnostic)
TMP_TOOL_CFG=$(mktemp --suffix=.yaml)
sed "s#http://localhost:8080/run_code#$SANDBOX_URL#" "$TOOL_CFG" > "$TMP_TOOL_CFG"

export PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_files="$DATA_DIR/test.parquet" \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESP \
    data.return_raw_chat=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TMP_TOOL_CFG" \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMP \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.reward_manager=naive \
    custom_reward_function.path=recipe/sod/reward.py \
    custom_reward_function.name=compute_score \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.nnodes=1 \
    "$@"
echo "=== SOD BENCHMARK EVAL DONE (avg@${VAL_N}; metrics under val-core/aime2024 & val-core/aime2025) ==="
