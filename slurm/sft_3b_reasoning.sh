#!/bin/bash
# SDAR-native SFT (verl fsdp_sft_trainer) on GiGPO-teacher reasoning trajectories. NEW, additive.
# Single-turn {prompt,response} (matches the per-turn eval); loss masked to response. Qwen2.5-3B,
# TCOD-v1 Wenbo-aligned (max_prompt 2048, lr 1e-5, train_batch 64 ~ OPD@250). gpu:2 FSDP+ulysses_sp2.
#SBATCH --job-name=sdar_sft3b_reason
#SBATCH --partition=zhanglab.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/pengchx3/text-dna/SDAR/slurm/%x-%j.out
#SBATCH --error=/home/pengchx3/text-dna/SDAR/slurm/%x-%j.err
set -x
ENV=/home/pengchx3/miniconda/envs/sdar
export PATH="$ENV/bin:$PATH"; unset ROCR_VISIBLE_DEVICES
export HF_HOME=/home/pengchx3/.cache/hf_sdar HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=$HF_HOME/hub TRANSFORMERS_CACHE=$HF_HOME/transformers; mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
LOCAL=/tmp/sdarsft_${SLURM_JOB_ID}; mkdir -p "$LOCAL"; trap 'rm -rf "$LOCAL"' EXIT
export TMPDIR="$LOCAL" RAY_TMPDIR="$LOCAL/ray" XDG_CACHE_HOME="$LOCAL/xdg" TRITON_CACHE_DIR="$LOCAL/triton" TORCHINDUCTOR_CACHE_DIR="$LOCAL/ind"
export VLLM_DO_NOT_TRACK=1 WANDB_MODE=offline
cd /home/pengchx3/text-dna/SDAR
SSD=/srv/disk00/sshfs/pengchx3; ls "$SSD/multi-turn-opd" >/dev/null 2>&1 || SSD=/tmp/galaxy_srv_disk00/pengchx3
DATA=$SSD/multi-turn-opd/data/alfworld_data/sft_reasoning_parquet
STUDENT=${STUDENT:-$SSD/multi-turn-opd/models/Qwen2.5-3B-Instruct}
SAVE=${SAVE:-$SSD/multi-turn-opd/checkpoints/SDAR_SFT/sdar_sft3b_reason}; mkdir -p "$SAVE"
STEPS=${STEPS:-250}; NGPUS=${NGPUS:-2}
[ -f "$DATA/train.parquet" ] || { echo "FATAL missing $DATA/train.parquet"; exit 1; }
echo "=== SDAR SFT: student=$STUDENT data=$DATA steps=$STEPS gpus=$NGPUS save=$SAVE ==="
torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
  -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=[] \
    data.response_dict_keys=[] \
    data.max_length=2560 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=1e-5 \
    model.partial_pretrain=$STUDENT \
    model.enable_gradient_checkpointing=True \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.default_local_dir=$SAVE \
    trainer.project_name=sdar_sft \
    trainer.experiment_name=sdar_sft3b_reason \
    trainer.logger=['console'] \
    trainer.total_training_steps=$STEPS \
    trainer.default_hdfs_dir=null
echo "DONE rc=$? -> $SAVE"
