#!/usr/bin/env python
"""Aggregate the sharded SDAR SOD math_tool eval into the overall avg@N (success_rate).

Each shard log holds per-source success rates for its 5 problems/source, e.g.
  'val/aime2024_success_rate': np.float64(0.4)
Since shards are an equal-size partition (5 problems/source each), the overall per-source
avg@N is the equal-weight mean across shards.

Usage: python recipe/sod/aggregate_sharded.py [log_dir]
"""
import glob
import os
import re
import sys

LOGD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "logs")
SOURCES = ("aime2024", "aime2025")


def extract(txt, source):
    # Prefer the FINAL compact summary line SDAR prints at the end (authoritative + always
    # present for both sources); the metrics-dict `np.float64(...)` print is unreliable (it can
    # omit one source's line, which silently mixed formats across shards). Use the LAST match.
    for pat in (rf"val/{source}_success_rate\s*:\s*([0-9.eE+\-]+)",
                rf"val/{source}_success_rate'\s*:\s*np\.float64\(([0-9.eE+\-]+)\)",
                rf"val/{source}/test_score\s*[:=]\s*([0-9.eE+\-]+)"):
        m = re.findall(pat, txt)
        if m:
            return float(m[-1])
    return None


def main():
    outs = sorted(glob.glob(os.path.join(LOGD, "sdar-shard*.out")))
    print(f"shard logs: {len(outs)}")
    agg = {s: [] for s in SOURCES}
    for out in outs:
        sh = re.search(r"sdar-shard(\d+)", out).group(1)
        txt = open(out, errors="ignore").read()
        row = {}
        for s in SOURCES:
            v = extract(txt, s)
            if v is not None:
                agg[s].append(v)
                row[s] = round(v * 100, 1)
        print(f"  shard{sh}: {row or 'PENDING'}")
    print("\n=== OVERALL avg@N success_rate (equal-weight mean across shards) ===")
    K = int(os.environ.get("K", "6"))
    overall = {}
    for s in SOURCES:
        v = agg[s]
        if v:
            overall[s] = sum(v) / len(v) * 100
            print(f"  {s}: {overall[s]:.2f}%  (from {len(v)}/{K} shards)")
        else:
            print(f"  {s}: --- (no shard finished)")
    complete = all(len(agg[s]) == K for s in SOURCES)
    print(f"\nPAPER: AIME2024=50.83  AIME2025=41.72 | standalone-SOD top_k=20: 48.12 / 36.77")
    if complete:
        print(f"SDAR math_tool (avg@N): AIME2024={overall['aime2024']:.2f}  AIME2025={overall['aime2025']:.2f}")
    else:
        got = {s: len(agg[s]) for s in SOURCES}
        print(f"[PARTIAL] headline withheld until all {K} shards done (have {got}); re-run run_sharded.sh to resume")


if __name__ == "__main__":
    main()
