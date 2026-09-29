"""Step 14 -- Burn-SEVERITY stratified recovery on DEPLOYED full-scene outputs.

Codex lever #2. Unlike scripts/10_eval_recovery.py (which re-tracks with the
emission cap lifted to max_gap=90, masking official-vs-union), this scores the
ALREADY-COMPUTED full-scene synth MOT outputs in their deployed configs, so the
official(g12) vs union vs dumb-coast difference is preserved. Reuses recovery_seq.

Buckets (m_glare observability): partial 0.3-0.7 / heavy 0.1-0.3 / total <0.1.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.data.splits import make_splits
from glaremot.eval.eval_recovery import (
    BUCKETS, _blank_stats, _add, _rate, load_events, load_mot, recovery_seq,
)

GLARE_ROOT = "D:/datasets/GlareMOT-Synth"
EVAL_ROOT = "results/eval_synth/trackers"

TRACKERS = [
    ("gt_official_synth_val", "official(g12)"),
    ("gt_union_r09_synth_val", "union(r0.9)"),
    ("gt_longcoast30_synth_val", "dumb30"),
    ("gt_longcoast90_synth_val", "dumb90"),
]


def main():
    sp = make_splits()
    seqs = [s for s in sp["val"]
            if os.path.isdir(os.path.join(GLARE_ROOT, s, "images"))
            and load_events(s, glare_root=GLARE_ROOT)]
    gt_cache = {s: load_mot(os.path.join(GLARE_ROOT, s, "mot", "gt", "gt.txt")) for s in seqs}

    print(f"\nSeverity-stratified recovery, DEPLOYED full-scene configs, {len(seqs)} synth val seqs.")
    print("Rcv@.5 = frac of burnt-target GT boxes covered; ID-Rcv = covered w/ entering id.\n")
    order = ["partial", "heavy", "total", "all"]
    rng = {"partial": "0.3-0.7", "heavy": "0.1-0.3", "total": "<0.10", "all": "<0.70"}

    aggs = {}
    for tdir, lab in TRACKERS:
        agg = _blank_stats()
        for s in seqs:
            trk = load_mot(os.path.join(EVAL_ROOT, tdir, "data", f"{s}.txt"))
            st = recovery_seq(s, gt_cache[s], trk, 0.5, glare_root=GLARE_ROOT)
            if st:
                _add(agg, st)
        aggs[tdir] = agg

    # Rcv@.5 table
    hdr = f"{'bucket':<9}{'m_glare':<10}{'n':>6}"
    for _t, lab in TRACKERS:
        hdr += f"{lab:>15}"
    print("RECOVERY-RECALL @ IoU0.5"); print(hdr); print("-" * len(hdr))
    for b in order:
        n = aggs[TRACKERS[0][0]][b]["n"]
        line = f"{b:<9}{rng[b]:<10}{n:>6}"
        for tdir, _lab in TRACKERS:
            line += f"{_rate(aggs[tdir][b]):>14.1f}%"
        print(line)

    print("\nID-RECOVERY-RECALL @ IoU0.5"); print(hdr); print("-" * len(hdr))
    for b in order:
        n = aggs[TRACKERS[0][0]][b]["n"]
        line = f"{b:<9}{rng[b]:<10}{n:>6}"
        for tdir, _lab in TRACKERS:
            line += f"{_rate(aggs[tdir][b], 'idrcv'):>14.1f}%"
        print(line)

    print("\nID continuity (episodes): IDSW / Frag / survived-to-end")
    for tdir, lab in TRACKERS:
        a = aggs[tdir]
        surv = f"{a['n_surv']}/{a['n_ep']}"
        print(f"  {lab:<14} IDSW={a['idsw']:>3}  Frag={a['frag']:>3}  surv={surv}")


if __name__ == "__main__":
    main()
