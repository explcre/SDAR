#!/bin/bash
# SDAR 7B ALFWorld from-scratch TRAINING reproduction on laniakea (8x 6000Ada).
# Reuses the parameterized run_sdar_3b_alfworld_repro.sh trainer (model-agnostic).
# Submit (auto-start when laniakea eval frees): sbatch --dependency=afterany:<seed7job> --nodelist=laniakea slurm/train_sdar_7b_alfworld_laniakea.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=420000M
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
ulimit -u $(ulimit -Hu) 2>/dev/null || true
# --- storage policy (see announcement_of_voyager_laniakea_ssd.md) ---
# caches: node-local /tmp (fast, no quota, no network load) -- NOT /home (over quota -> EDQUOT hang) NOR sshfs
LOCAL=/tmp/sdar_${SLURM_JOB_ID}; mkdir -p "$LOCAL"
export TMPDIR="$LOCAL" TMP="$LOCAL" TEMP="$LOCAL" RAY_TMPDIR="$LOCAL"
export XDG_CACHE_HOME="$LOCAL/xdg" TRITON_CACHE_DIR="$LOCAL/triton" VLLM_CACHE_ROOT="$LOCAL/vllm"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/inductor" TORCH_EXTENSIONS_DIR="$LOCAL/torchext"
export VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
# checkpoints: shared SSD reachable from galaxy/laniakea/voyager (self-heal mount; fallback node-local)
SSD=/tmp/galaxy_srv_disk00/pengchx3
if ! ls "$SSD" >/dev/null 2>&1; then
  echo "SSD not mounted; attempting sshfs remount..."; mkdir -p "$SSD"
  sshfs -o allow_other,default_permissions,reconnect pengchx3@galaxy.ics.uci.edu:/srv/disk00/sshfs/pengchx3 "$SSD" 2>&1 | head -3
fi
if ls "$SSD" >/dev/null 2>&1 && touch "$SSD/.wt_$$" 2>/dev/null; then rm -f "$SSD/.wt_$$"; echo "SSD OK: $SSD"; else echo "SSD UNAVAILABLE -> checkpoints to node-local $LOCAL"; SSD="$LOCAL"; fi
mkdir -p "$SSD/sdar_train"
unset ROCR_VISIBLE_DEVICES
echo "JOB=$SLURM_JOB_ID NODE=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

cd /home/pengchx3/text-dna/SDAR
export VAL_BEFORE_TRAIN=False   # skip the slow/hang-prone initial 140-game eval; trained-model success comes from test_freq during training
export NGPUS=4 TP=2 GMU=${GMU:-0.5} \
  MODEL_PATH=/home/pengchx3/sdar_models/Qwen2.5-7B-Instruct \
  EXPNAME=sdar_qwen2.5_7b_repro \
  OUTDIR=$SSD/sdar_train/sdar_qwen2.5_7b_repro
bash recipe/hgpo/run_sdar_3b_alfworld_repro.sh vllm
echo "DONE $(date)"
