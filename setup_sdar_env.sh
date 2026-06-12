#!/bin/bash
# Build the `sdar` conda env for evaluating GiGPO-7B on ALFWorld (val_only).
# flash-attn is skipped first (val_only eval uses vLLM kernels); add later only if needed.
set -e
source /home/pengchx3/miniconda/etc/profile.d/conda.sh
export HF_HOME=/tmp/galaxy_srv_disk00/pengchx3/hf_cache_sdar
echo "=== create env ==="
conda create -n sdar python=3.12 -y
SD=/home/pengchx3/miniconda/envs/sdar/bin
echo "=== vllm 0.11.0 (pins torch 2.8) ==="
$SD/pip install --no-cache-dir vllm==0.11.0
echo "=== verl (SDAR) editable, minus flash-attn extras ==="
$SD/pip install --no-cache-dir -e /home/pengchx3/text-dna/SDAR
echo "=== alfworld + env deps ==="
$SD/pip install --no-cache-dir gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
echo "=== pin transformers per requirements ==="
$SD/pip install --no-cache-dir "transformers==4.51.1"
echo "=== sanity: imports ==="
$SD/python -c "import vllm, verl, alfworld, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| OK')"
echo "SDAR_ENV_SETUP_DONE"
