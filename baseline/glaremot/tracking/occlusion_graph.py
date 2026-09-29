"""Depth-ordering & occlusion-graph (spec v4 §7, T0 pixel-space).

Turns person-person occlusion from an unpredictable failure into a *predictable*
geometric signal. For a static camera, the image-y of a foot point is a monotone
proxy for depth: larger foot-y = lower in the frame = NEARER the camera = the
occluder. We compare tracks pairwise; when their predicted boxes overlap and they
sit at clearly different depths, the FARTHER track loses visibility proportional to
how much of its box the nearer one covers.

The output per track is ``vis_geom in [0,1]`` = the fraction NOT hidden by a nearer
person. It multiplies ``m_glare`` (photometric) to form the unified reliability
``m`` used by the controller's gating (spec v4 §4).

This module is PURE LOGIC (numpy only) and unit-tested without any data / model:
  - :func:`compute_vis_geom` is the functional core (boxes + depth keys -> vis_geom).
  - :func:`build_occlusion_graph` is the object wrapper used by the controller; it
    reads each track's predicted box + depth cue and writes ``trk.vis_geom``.

T1 swap: replace the pixel ``foot_y`` depth key with metric ``s`` from the homography
(``geometry/homography_bev.py``) - same interface, one function changes (spec v4 §7).
"""
from itertools import combinations

import numpy as np

__all__ = [
    "box_iou_xyxy", "covered_fraction", "depth_key", "track_depth_key",
    "compute_vis_geom", "build_occlusion_graph",
]


def _as_xyxy(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _area(box):
    x1, y1, x2, y2 = _as_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a, b):
    ax1, ay1, ax2, ay2 = _as_xyxy(a)
    bx1, by1, bx2, by2 = _as_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou_xyxy(a, b):
    """IoU of two pixel boxes (x1,y1,x2,y2)."""
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def covered_fraction(far_box, near_box):
    """Fraction of ``far_box``'s area covered by ``near_box`` (in [0,1]).

    This is the share of the farther person's box hidden behind the nearer one,
    so ``vis_geom`` of the far track is reduced by exactly this much.
    """
    fa = _area(far_box)
    if fa <= 0:
        return 0.0
    return float(np.clip(_intersection(far_box, near_box) / fa, 0.0, 1.0))


def depth_key(box_xyxy=None, foot_y=None, foot_visible=False,
              beacon_present=False, beacon_mu_y=None, h_prior=0.0,
              k_head2foot=1.0):
    """Monotone depth proxy (spec v4 §7). LARGER = NEARER the camera = occluder.

    Cue priority (use the best available):
      1. clean foot point ``foot_y`` (when ``foot_visible``);
      2. head beacon: ``beacon_mu_y + k_head2foot * h_prior`` (head -> approx foot,
         so we can order depth in the dark/glare WITHOUT seeing feet);
      3. fallback: bottom edge of the predicted box (``y2``).
    """
    if foot_visible and foot_y is not None:
        return float(foot_y)
    if beacon_present and beacon_mu_y is not None:
        return float(beacon_mu_y) + float(k_head2foot) * float(h_prior)
    if box_xyxy is not None:
        return _as_xyxy(box_xyxy)[3]          # bottom_y
    raise ValueError("depth_key needs a foot, a beacon, or a box")


def track_depth_key(trk, k_head2foot=1.0):
    """depth_key for a duck-typed track object used by :func:`build_occlusion_graph`.

    Reads (all optional): ``predbox_img`` (xyxy), ``foot_visible``, ``foot_y``,
    ``h_prior``, and ``beacon`` with ``.present`` / ``.mu_y`` (or a ``beacon`` dict
    with those keys).
    """
    beacon = getattr(trk, "beacon", None)
    if isinstance(beacon, dict):
        b_present = bool(beacon.get("present", False))
        b_mu_y = beacon.get("mu_y", None)
    else:
        b_present = bool(getattr(beacon, "present", False)) if beacon is not None else False
        b_mu_y = getattr(beacon, "mu_y", None) if beacon is not None else None
    return depth_key(
        box_xyxy=getattr(trk, "predbox_img", None),
        foot_y=getattr(trk, "foot_y", None),
        foot_visible=bool(getattr(trk, "foot_visible", False)),
        beacon_present=b_present, beacon_mu_y=b_mu_y,
        h_prior=float(getattr(trk, "h_prior", 0.0)),
        k_head2foot=k_head2foot,
    )


def compute_vis_geom(boxes, depth_keys, tau_ov=0.2, dmin=8.0):
    """Functional core: per-box geometric visibility from pairwise occlusion.

    Args:
        boxes:      sequence of N pixel boxes (x1,y1,x2,y2).
        depth_keys: sequence of N depth proxies (larger = nearer; see :func:`depth_key`).
        tau_ov:     min IoU of predicted boxes to consider an occlusion.
        dmin:       min depth-key separation to trust an ordering (avoids near-ties).

    Returns:
        ``vis_geom`` float array of shape (N,) in [0,1]; 1.0 = fully visible.
    """
    boxes = list(boxes)
    depth_keys = [float(d) for d in depth_keys]
    n = len(boxes)
    if n != len(depth_keys):
        raise ValueError("boxes and depth_keys must have the same length")
    vis = np.ones(n, np.float32)
    for i, j in combinations(range(n), 2):
        if box_iou_xyxy(boxes[i], boxes[j]) <= tau_ov:
            continue
        if abs(depth_keys[i] - depth_keys[j]) <= dmin:    # near-tie -> no hard occlusion
            continue
        near, far = (i, j) if depth_keys[i] > depth_keys[j] else (j, i)
        vis[far] *= (1.0 - covered_fraction(boxes[far], boxes[near]))
    return np.clip(vis, 0.0, 1.0)


def build_occlusion_graph(tracks, tau_ov=0.2, dmin=8.0, k_head2foot=1.0):
    """Object wrapper (spec v4 §7 pseudocode). Sets ``trk.vis_geom`` on each track.

    Each track must expose ``predbox_img`` (xyxy) and a depth cue readable by
    :func:`track_depth_key`. Returns the same ``tracks`` list for chaining.
    """
    tracks = list(tracks)
    if not tracks:
        return tracks
    boxes = [t.predbox_img for t in tracks]
    keys = [track_depth_key(t, k_head2foot=k_head2foot) for t in tracks]
    vis = compute_vis_geom(boxes, keys, tau_ov=tau_ov, dmin=dmin)
    for t, v in zip(tracks, vis):
        t.vis_geom = float(v)
    return tracks
