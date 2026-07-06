#!/bin/bash
# Resumable + parallel sharded SOD math_tool eval on SDAR. Splits the 60 problems into K
# shards (5 aime2024 + 5 aime2025 each), runs each as an independent slurm_sod_eval.sh on
# gpu:1 across galaxy/laniakea/voyager. Re-running SKIPS shards whose log already has the
# success_rate (resume) and skips shards currently running. Aggregate with aggregate_sharded.py.
#   K=6 NODES="galaxy laniakea voyager" bash recipe/sod/run_sharded.sh
set -u
W=/home/pengchx3/text-dna/SDAR-sod; cd "$W"
module load slurm 2>/dev/null || true; export PATH="/pkg/slurm/22.05.3/bin:$PATH"
SSD=/tmp/galaxy_srv_disk00/pengchx3; [ -d "$SSD/SOD" ] || SSD=/srv/disk00/sshfs/pengchx3
SHARDROOT=$SSD/sdar_sod_data_shards
LOGD=$W/logs; mkdir -p "$LOGD"
K=${K:-6}
MODEL=${MODEL:-$SSD/SOD/SOD-1.7B}
read -r -a NODES <<< "${NODES:-galaxy laniakea voyager galaxy laniakea voyager}"
# faithful SOD sampling by default (parallelism makes 20480/16 tractable); override to bound
VAL_N=${VAL_N:-32}; MAX_RESP=${MAX_RESP:-20480}; MAX_TURNS=${MAX_TURNS:-16}
MAX_PROMPT=${MAX_PROMPT:-16384}; MAX_BATCHED=${MAX_BATCHED:-40960}

done=0; running=0; submitted=0
for i in $(seq 0 $((K-1))); do
  out=$LOGD/sdar-shard${i}.out
  if grep -q "aime2024_success_rate\|aime2025_success_rate" "$out" 2>/dev/null; then echo "shard$i: DONE -> skip"; done=$((done+1)); continue; fi
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "sdar-shard${i}"; then echo "shard$i: running/queued -> skip"; running=$((running+1)); continue; fi
  N=${NODES[$((i % ${#NODES[@]}))]}
  # per-shard problem count (5+5=10 for K=6, 60 problems)
  NP=$(( 60 / K )); [ "$NP" -lt 1 ] && NP=1
  JID=$(sbatch --parsable --job-name=sdar-shard${i} --gres=gpu:1 --nodelist="$N" \
    --cpus-per-task=8 --mem=64000M \
    --output="$out" --error="$LOGD/sdar-shard${i}.err" \
    --export=ALL,MODEL_PATH=$MODEL,DATA_DIR=$SHARDROOT/shard${i},N_PROBLEMS=$NP,VAL_N=$VAL_N,MAX_PROMPT=$MAX_PROMPT,MAX_RESP=$MAX_RESP,MAX_TURNS=$MAX_TURNS,MAX_BATCHED=$MAX_BATCHED,HIST_LEN=8,TRUNCATION=left,NATIVE_MULTITURN=${NATIVE_MULTITURN:-True} \
    recipe/sod/slurm_sod_eval.sh 2>&1)
  echo "shard$i -> $N jid=$JID (N_PROBLEMS=$NP)"; submitted=$((submitted+1))
done
echo "=== summary: done=$done running=$running submitted=$submitted (re-run to resume) ==="
