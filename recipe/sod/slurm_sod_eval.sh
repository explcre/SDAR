#!/bin/bash
# SLURM wrapper: run the SOD math_tool benchmark eval in the SDAR (sdar) env, starting a
# SandboxFusion server via singularity. Node-aware SSD paths. Smoke defaults; override env vars.
#SBATCH --job-name=sod-sdar-smoke
#SBATCH --partition=zhanglab.p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000M
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/pengchx3/text-dna/SDAR-sod/logs/%x-%j.out
#SBATCH --error=/home/pengchx3/text-dna/SDAR-sod/logs/%x-%j.err
set -x
W=/home/pengchx3/text-dna/SDAR-sod; cd "$W"; mkdir -p logs
# node-aware SSD prefix (galaxy: /srv local; laniakea/voyager: /tmp sshfs)
if ls "/tmp/galaxy_srv_disk00/$(whoami)/SOD" >/dev/null 2>&1; then SSD="/tmp/galaxy_srv_disk00/$(whoami)"; else SSD="/srv/disk00/sshfs/$(whoami)"; fi
echo "node=$(hostname) SSD=$SSD"
ENV=/home/pengchx3/miniconda/envs/sdar
export PYTHONNOUSERSITE=1 PATH="$ENV/bin:$PATH" PYTHONPATH="$W:${PYTHONPATH:-}"
# HF caches -> /home (the .bashrc pins them to galaxy /srv, unwritable off-galaxy)
export HF_HOME=/home/pengchx3/.cache/hf_tcod HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_MODE=offline
unset ROCR_VISIBLE_DEVICES

export MODEL_PATH=${MODEL_PATH:-$SSD/multi-turn-opd/models/Qwen2.5-3B-Instruct}
export DATA_DIR=${DATA_DIR:-$SSD/sdar_sod_data_smoke}
export SANDBOX_SIF=${SANDBOX_SIF:-$SSD/SOD/sandbox/sandbox_fusion.sif}
export SANDBOX_URL=${SANDBOX_URL:-http://localhost:8080/run_code}
export NGPUS=1 INFER_TP=1
export N_PROBLEMS=${N_PROBLEMS:-4} VAL_N=${VAL_N:-2}
export MAX_RESP=${MAX_RESP:-8192} MAX_TURNS=${MAX_TURNS:-8} HIST_LEN=${HIST_LEN:-8}
echo "model=$MODEL_PATH data=$DATA_DIR N_PROBLEMS=$N_PROBLEMS VAL_N=$VAL_N"
bash recipe/sod/run_sod_eval.sh
echo "=== SLURM SOD-SDAR EVAL WRAPPER DONE ==="
