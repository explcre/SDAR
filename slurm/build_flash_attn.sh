#!/bin/bash
# Source-build flash-attn for the sdar env (torch 2.8.0+cu128, py3.12, GLIBC 2.31).
# Prebuilt wheels need GLIBC 2.32+, so we compile against local CUDA 12.4. CPU-only build.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=480000M
#SBATCH --partition=zhanglab.p
#SBATCH --time=12:00:00
#SBATCH --output=/home/pengchx3/text-dna/SDAR/slurm/%x-%j.out
set -e
ENV=/home/pengchx3/miniconda/envs/sdar
export PATH="$ENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=6   # flash-attn nvcc procs use ~8-12GB each; 6 x ~12GB fits in 480G (32 OOM'd)
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"   # 3090, 6000Ada, H100
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TMPDIR=/dev/shm/fabuild_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
echo "NODE=$(hostname) nvcc=$(nvcc --version | grep release)"
$ENV/bin/pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3
echo "=== verify ==="
$ENV/bin/python -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
echo "FLASH_ATTN_BUILD_DONE"
