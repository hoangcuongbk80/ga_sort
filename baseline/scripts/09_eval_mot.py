"""Step 09 - MOT benchmark (HOTA / IDF1 / MOTA via TrackEval).

Examples:
  python scripts/09_eval_mot.py --split test                  # 12 real-glare seqs
  python scripts/09_eval_mot.py --split all --resume          # full benchmark
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glaremot.eval.eval_mot import main

if __name__ == "__main__":
    main()
