"""Step 03 - Synthesize headlamp glare on the q-sequences (optional re-run).

Turns a CLEAN tracking sequence into a paired "burnt" sequence: a person's
headlamp glare rises smoothly (raised-cosine envelope, <= 3 s), holds - fully burning the person, total feature loss - then fades. Rendered in linear
light (core + bloom halo + starburst), with the bloom sized to cover the whole
person. MOT labels are copied UNCHANGED (amodal GT stays valid), and
``meta/glare_schedule.json`` records the events (the forecaster reads it to
reconstruct the per-frame confidence drop).

If the glare dataset already exists you can skip this step entirely.

Run:  python scripts/03_synth_glare.py [--seq <name>] [--preview]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.perception.synth_glare import main

if __name__ == "__main__":
    main()
