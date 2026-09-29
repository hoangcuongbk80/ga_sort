"""Step 04 - Synthesize person-person occlusion GT (optional re-run).

Glare erases a person's OWN pixels; occlusion erases them because a NEARER
person stands in front. This step composites a donor track over a target track
of the same sequence (re-timed + translated so the donor's foot-y is larger =
nearer = the occluder) and records ``vis_geom_gt`` - the fraction of each
amodal box NOT hidden - using the SAME occlusion-graph code the tracker runs
online, so train == runtime by construction.

Use ``--no-images`` to write only the extended GT text (fast, HDD-light): the
forecaster needs only the trajectories + vis_geom, not the pixels.

Run:  python scripts/04_compose_occlusion.py --no-images
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.perception.compose_occlusion import main

if __name__ == "__main__":
    main()
