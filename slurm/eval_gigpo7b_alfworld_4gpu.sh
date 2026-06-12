#!/bin/bash
# Eval GiGPO-Qwen2.5-7B on ALFWorld via SDAR (verl-agent val_only), default setup.
# Submit: sbatch --job-name=sdar_gigpo7b_seen --export=ALL,VAL_OUT=True  slurm/eval_gigpo7b_alfworld.sh
#         sbatch --job-name=sdar_gigpo7b_unseen --export=ALL,VAL_OUT=False slurm/eval_gigpo7b_alfworld.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=200000M
#SBATCH --partition=zhanglab.p
#SBATCH --time=30-8:00:00
#SBATCH --output=/home/pengchx3/text-dna/SDAR/slurm/%x-%j.out
set -u
ENV=/home/pengchx3/miniconda/envs/sdar
SSD=/tmp/galaxy_srv_disk00/pengchx3
export PATH="$ENV/bin:$PATH"
export HF_HOME=/home/pengchx3/.cache/hf_sdar
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/transformers HF_ASSETS_CACHE=$HF_HOME/assets
mkdir -p $HF_HUB_CACHE $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $HF_ASSETS_CACHE
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA=/home/pengchx3/alfworld_data  # NFS copy; config_tw.yaml reads $ALFWORLD_DATA/json_2.1.1/*. unset -> env init hangs
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# (P2P/IB-disable removed: the prior "multi-GPU hang" was the unset ALFWORLD_DATA, not NCCL; intra-node TP=4 wants P2P on)
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/dev/shm/sdar_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; export TMP="$TMPDIR" TEMP="$TMPDIR"
export RAY_TMPDIR=$TMPDIR
unset ROCR_VISIBLE_DEVICES
# pin to the free GPUs SLURM gave us (cgroup already limits visibility)
echo "JOB=$SLURM_JOB_ID NODE=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES VAL_OUT=${VAL_OUT:-True}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

$ENV/bin/ray stop --force >/dev/null 2>&1 || true; sleep 3  # clean lingering Ray from prior crashed jobs
cd /home/pengchx3/text-dna/SDAR
export VAL_OUT=${VAL_OUT:-True} SEED=${SEED:-123} NGPUS=4 TP=4 GMU=${GMU:-0.6}
bash recipe/hgpo/run_gigpo7b_alfworld_eval.sh vllm
echo "DONE $(date)"
