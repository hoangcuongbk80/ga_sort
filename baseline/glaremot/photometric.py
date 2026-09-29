"""Domain-invariant photometric primitives for the glare beacon (patch v2.1 Fix 1).

The lamp core is the SAME photometric signature on real and synthetic frames:
near-clipped luminance, low chroma, compact shape. Computing it analytically here
(instead of relying on the synthetic paint mask) makes the beacon transfer to real
video. Used both as the Stage-1 GT mask (data/beacon_dataset.py) and at inference
(controller.py) so train/test see an identical target.
"""
import cv2
import numpy as np


def _eccentricity(mask):
    """Eccentricity in [0,1] from 2nd-order central moments. 0 = round, ->1 = line."""
    ys, xs = np.nonzero(mask)
    if xs.size < 5:
        return 0.0
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    mu20 = ((x - xm) ** 2).mean()
    mu02 = ((y - ym) ** 2).mean()
    mu11 = ((x - xm) * (y - ym)).mean()
    common = np.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * mu11 ** 2))
    l1 = (mu20 + mu02 + common) / 2.0
    l2 = (mu20 + mu02 - common) / 2.0
    if l1 <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - l2 / l1)))


def saturation_map(bgr, v_thr=0.92, s_max=0.35):
    """Raw per-pixel washed-out/clipped map `sat(p)` (spec v4 stage1 §1.3): bright
    (near-clip luminance) AND low chroma. Boolean HxW. This is the UNFILTERED
    primitive (no compactness test) used to measure how much of a body box is
    photometrically destroyed; `saturated_core_mask` adds component filtering on top
    to isolate the compact lamp core."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    V, S = hsv[..., 2], hsv[..., 1]
    return (V >= v_thr) & (S <= s_max)


def saturated_core_mask(bgr, v_thr=0.92, s_max=0.35, min_area=12,
                        ecc_max=0.85, fill_min=0.55, open_ksize=3):
    """Compact saturated-core (lamp) mask. Bright + washed-out + compact components
    only -> rejects elongated beams / radial starburst streaks (patch v2.1 Fix 1.1).
    Returns a uint8 {0,1} mask the same HxW as `bgr`."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    V, S = hsv[..., 2], hsv[..., 1]
    core = ((V >= v_thr) & (S <= s_max)).astype(np.uint8)
    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    out = np.zeros_like(core)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area / float(max(1, w * h)) < fill_min:       # not compact -> beam/streak
            continue
        comp = labels == i
        if _eccentricity(comp) >= ecc_max:                # elongated -> beam/streak
            continue
        out[comp] = 1
    return out


def box_area(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def mask_area_inside(M, xyxy):
    """Count of mask-on pixels inside the (pixel) box xyxy."""
    H, W = M.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(M[y1:y2, x1:x2].sum())


def core_centroid_in_box(M, xyxy):
    """Centroid (px, image coords) of the largest core component inside the box, or
    None. Gives a domain-invariant beacon point that does not depend on the head's
    learned mu (which was trained on synthetic data)."""
    H, W = M.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    sub = M[y1:y2, x1:x2].astype(np.uint8)
    if sub.sum() == 0:
        return None
    n, _, stats, cents = cv2.connectedComponentsWithStats(sub, 8)
    if n <= 1:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[j]
    return np.array([cx + x1, cy + y1], np.float32)
