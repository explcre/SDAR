#!/usr/bin/env python
"""Split the SOD math_tool eval parquet (test.parquet, 60 problems) into K shards for
parallel sharded evaluation. Each shard keeps an equal split PER data_source (aime2024/
aime2025), so per-source avg@N is unchanged; only the problem set is partitioned. Aggregate
the per-shard success_rate with an equal-weight mean (aggregate_sharded.py).

Usage: python recipe/sod/shard_data.py <in_dir> <out_dir> <K>
  -> writes <out_dir>/shard{i}/test.parquet  for i in 0..K-1
"""
import os
import sys

import pandas as pd


def main(in_dir: str, out_dir: str, k: int) -> None:
    df = pd.read_parquet(os.path.join(in_dir, "test.parquet")).reset_index(drop=True)
    # contiguous, equal-per-source shards
    shards = {i: [] for i in range(k)}
    for ds, sub in df.groupby("data_source"):
        sub = sub.reset_index(drop=True)
        n = len(sub)
        bounds = [round(i * n / k) for i in range(k + 1)]
        for i in range(k):
            shards[i].append(sub.iloc[bounds[i]:bounds[i + 1]])
    for i in range(k):
        d = os.path.join(out_dir, f"shard{i}")
        os.makedirs(d, exist_ok=True)
        s = pd.concat(shards[i]).reset_index(drop=True)
        s.to_parquet(os.path.join(d, "test.parquet"), index=False)
        cnt = s["data_source"].value_counts().to_dict()
        print(f"  shard{i}: {len(s)} problems {cnt} -> {d}/test.parquet")
    print(f"wrote {k} shards under {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
