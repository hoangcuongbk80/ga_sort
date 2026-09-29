"""Step 08 - Run the tracker on a sequence or a raw video (annotated mp4 out).

Examples:
  python scripts/08_run_tracker.py --video path/to/clip.mp4
  python scripts/08_run_tracker.py --seq 182_sequence_61q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.tracking.tracker import main

if __name__ == "__main__":
    main()
