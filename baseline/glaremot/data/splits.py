"""Frozen sequence-level train/val/test split (80/17/20 sequences).

The split is stored in ``splits.json`` next to this module and is the SINGLE
split used by every stage (detector, h_psi, tracker benchmark):

* ``test``  - the 20 REAL-glare sequences (people facing the camera, headlamp
  burn covering the whole body). These are the deployment target and are held
  out of ALL training, including the detector.
* ``val``   - 17 sequences with the same distribution as train; used for
  checkpoint selection everywhere.
* ``train`` - the remaining 80 sequences.

Splitting is sequence-level on purpose: consecutive frames are near-identical,
so any frame-level split would leak train data into validation.
"""
import json
import os

_SPLIT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")


def make_splits():
    """Return {'train': [...], 'val': [...], 'test': [...]} sequence names."""
    with open(_SPLIT_JSON, encoding="utf-8") as f:
        sp = json.load(f)
    return {"train": list(sp["train"]), "val": list(sp["val"]), "test": list(sp["test"])}


def split_of(seq):
    """Which split a sequence belongs to ('train' | 'val' | 'test' | None)."""
    sp = make_splits()
    for k in ("train", "val", "test"):
        if seq in sp[k]:
            return k
    return None
