#!/bin/bash
W=/home/pengchx3/text-dna/SDAR-sod; cd "$W"; ENV=/home/pengchx3/miniconda/envs/sod
for it in $(seq 1 288); do
  n=$(grep -l "aime2024_success_rate\|aime2025_success_rate" logs/sdar-shard*.out 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M')] scored shards: $n/6"
  if [ "$n" -ge 6 ]; then
    echo "=== ALL 6 SCORED — aggregating ==="
    PYTHONNOUSERSITE=1 "$ENV/bin/python" recipe/sod/aggregate_sharded.py logs
    break
  fi
  sleep 300
done
