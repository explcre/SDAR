#!/bin/bash
# SDAR 3B ALFWorld from-scratch TRAINING reproduction on galaxy (3rd parallel track).
# Submit: sbatch --job-name=sdar3b_train --nodelist=galaxy slurm/train_sdar_3b_alfworld.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=160000M
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-08:00:00
#SBATCH --output=/home/pengchx3/text-dna/SDAR/slurm/%x-%j.out
set -u
ENV=/home/pengchx3/miniconda/envs/sdar
export PATH="$ENV/bin:$PATH"
export HF_HOME=/home/pengchx3/.cache/hf_sdar
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/transformers HF_ASSETS_CACHE=$HF_HOME/assets
mkdir -p $HF_HUB_CACHE $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $HF_ASSETS_CACHE
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA=/home/pengchx3/alfworld_data
export TOKENIZERS_PARALLELISM=false
ulimit -u $(ulimit -Hu) 2>/dev/null || true   # raise nproc to hard max (avoid Ray 'Resource temporarily unavailable')
export TMPDIR=/tmp/sdar_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; export TMP="$TMPDIR" TEMP="$TMPDIR"
export RAY_TMPDIR=$TMPDIR
unset ROCR_VISIBLE_DEVICES
echo "JOB=$SLURM_JOB_ID NODE=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

cd /home/pengchx3/text-dna/SDAR
export NGPUS=2 TP=2 GMU=${GMU:-0.5} \
  MODEL_PATH=/home/pengchx3/sdar_models/Qwen2.5-3B-Instruct \
  EXPNAME=sdar_qwen2.5_3b_repro \
  OUTDIR=/home/pengchx3/sdar_train/sdar_qwen2.5_3b_repro
bash recipe/hgpo/run_sdar_3b_alfworld_repro.sh vllm
echo "DONE $(date)"
