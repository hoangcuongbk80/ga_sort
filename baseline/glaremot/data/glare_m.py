r"""Derive per-(track, frame) ``m_glare`` from the synthetic-glare schedule
(spec v4 §4) - the low-`m` visibility signal the recovery benchmark scores through.

A synthetic-glare sequence has the SAME GT boxes as its clean source (synth only
paints pixels). The only thing glare contributes is the confidence channel ``m``
dropping while a person is burnt. We reconstruct exactly that drop from
``GlareMOT-Synth/<seq>/meta/glare_schedule.json`` using the SAME raised-cosine
envelope the renderer used (`synth_glare.env_value`):

    intensity_t = env_value(t-start, attack, hold, decay, peak)   in [0, peak]
    m_glare_t   = clip(1 - intensity_t, m_floor, 1)

so m_glare -> ~0 at the burn peak and ->1 at the edges. Frames are 0-indexed
(matches the gt / obb_tracks convention). No images are read.
"""
import os
import json

import numpy as np

from ..perception.synth_glare import env_value

from ..config import data_path, load_config
GLARE_ROOT = data_path(load_config(), "glare")
M_FLOOR = 0.02


def glare_m_map(seq_name, glare_root=GLARE_ROOT, m_floor=M_FLOOR):
    """{(target_id, frame0): m_glare} for one sequence, or {} if no schedule.

    Overlapping events keep the LOWEST m_glare (strongest burn) at each frame."""
    sched = os.path.join(glare_root, seq_name, "meta", "glare_schedule.json")
    if not os.path.exists(sched):
        return {}
    data = json.load(open(sched, encoding="utf-8"))
    out = {}
    for ev in data.get("events", []):
        tid = int(ev["target_id"])
        start = int(ev["start"])
        for k in range(int(ev["duration"])):
            f = start + k
            inten = env_value(k, ev["attack"], ev["hold"], ev["decay"], ev["peak"])
            mg = float(np.clip(1.0 - inten, m_floor, 1.0))
            key = (tid, f)
            if key not in out or mg < out[key]:
                out[key] = mg
    return out


if __name__ == "__main__":
    import sys
    seq = sys.argv[1] if len(sys.argv) > 1 else "125_sequence_12q"
    mp = glare_m_map(seq)
    print(f"{seq}: {len(mp)} (id,frame) glare entries")
    if mp:
        lo = min(mp.values())
        print(f"  min m_glare={lo:.3f}  sample:",
              sorted(mp.items())[:3])
