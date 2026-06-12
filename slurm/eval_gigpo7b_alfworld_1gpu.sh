#!/bin/bash
# 1-GPU variant: eval GiGPO-7B on ALFWorld (TP=1, higher gpu_mem_util to fit 7B on one card).
# Lets the job grab a single free GPU (e.g. galaxy 3090). Tight on 24GB; for ≥40GB it's easy.
# Submit: sbatch --job-name=sdar_gigpo7b_seen1g --export=ALL,VAL_OUT=True slurm/eval_gigpo7b_alfworld_1gpu.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160000M
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-08:00:00
#SBATCH --output=/home/pengchx3/text-dna/SDAR/slurm/%x-%j.out
set -u
ENV=/home/pengchx3/miniconda/envs/sdar
SSD=/tmp/galaxy_srv_disk00/pengchx3
export PATH="$ENV/bin:$PATH"
export HF_HOME=/home/pengchx3/.cache/hf_sdar
# pin ALL hf cache vars to /tmp (override any .bashrc leak to /srv, which is galaxy-only)
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/transformers HF_ASSETS_CACHE=$HF_HOME/assets
mkdir -p $HF_HUB_CACHE $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $HF_ASSETS_CACHE
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA=/home/pengchx3/alfworld_data  # NFS copy; config_tw.yaml reads $ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen}. unset -> env init hangs
export TOKENIZERS_PARALLELISM=false
# caches -> node-local /tmp (NOT /home: over-quota -> EDQUOT made cache writes block = the post-model-load HANG)
LOCAL=/tmp/sdar_eval_${SLURM_JOB_ID}; mkdir -p "$LOCAL"
export TMPDIR="$LOCAL" TMP="$LOCAL" TEMP="$LOCAL" RAY_TMPDIR="$LOCAL"
export XDG_CACHE_HOME="$LOCAL/xdg" TRITON_CACHE_DIR="$LOCAL/triton" VLLM_CACHE_ROOT="$LOCAL/vllm"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/inductor" TORCH_EXTENSIONS_DIR="$LOCAL/torchext"
export VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
unset ROCR_VISIBLE_DEVICES
echo "JOB=$SLURM_JOB_ID NODE=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES VAL_OUT=${VAL_OUT:-True}"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

cd /home/pengchx3/text-dna/SDAR
export VAL_OUT=${VAL_OUT:-True} SEED=${SEED:-123} NGPUS=1 TP=1 GMU=${GMU:-0.6}  # 0.6=recipe default; leaves room for verl FSDP actor+ref on the same card (0.85 OOM'd on 48G)
# GiGPO model now lives on SSD (deleted /home copy to free quota); recipe respects ${MODEL_PATH:-}/${OUTDIR:-}
export MODEL_PATH=${MODEL_PATH:-$SSD/multi-turn-opd/models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld}
export OUTDIR=${OUTDIR:-$SSD/sdar_eval/gigpo7b_valout${VAL_OUT}_seed${SEED}_h${HISTLEN:-4}}
echo "MODEL_PATH=$MODEL_PATH HISTLEN=${HISTLEN:-4}"
bash recipe/hgpo/run_gigpo7b_alfworld_eval.sh vllm
echo "DONE $(date)"
