"""Offline visibility-label generation (spec v4 §4, §9, §13) - T0, no model needed.

One source of truth for the reliability signal ``m`` and its provenance, computed
analytically from ground-truth boxes + a burn-mask + the occlusion-graph. Used to
build training targets and the extended MOT label columns (spec v4 §13):

    ... visibility, m_glare, vis_geom, foot_y, prov, beacon_present [, s, d]

Definition (spec v4 §4) - ``m`` is *localization* reliability ("can the detector box
be trusted to localize this person"), NOT "fraction the eye can see". Two independent
axes multiplied:

    m_glare  = how much of the BODY is photometrically intact (torso-weighted, so a
               burnt head barely lowers it - YOLO still localizes from torso/legs).
    vis_geom = fraction NOT hidden by a nearer person (from occlusion_graph §7).
    m        = m_glare * vis_geom

This module is PURE LOGIC (numpy only); unit-tested without any dataset/model.
"""
import numpy as np

__all__ = [
    "m_glare_from_burn", "combine_m", "classify_provenance",
    "visibility_band", "make_visibility_label",
    "BAND_THRESHOLDS", "PROV_TAU",
]

# Band -> (mode, human description). Thresholds are lower-inclusive (spec v4 §4 table).
BAND_THRESHOLDS = [
    (0.8, "A", "RELIABLE", "DETECTION"),     # full body, or only head/lamp burnt
    (0.4, "B", "DEGRADED", "DETECTION+SIZE"),  # lost upper half / partly occluded
    (0.1, "C", "BEACON-ONLY", "BEACON"),     # body mostly burnt but lamp core survives
    (-1.0, "D", "LOST", "FORECAST"),         # full burn no usable core, or fully occluded
]
PROV_TAU = 0.4   # default axis threshold for provenance classification


def _box_to_px(box_xyxy, W, H):
    """Accept a normalized (cx,cy,w,h) or pixel (x1,y1,x2,y2) box and return int
    pixel corners clamped to the frame. Heuristic: values <= ~1.5 are treated as
    normalized cx,cy,w,h (matches the dataset's YOLO label convention)."""
    b = [float(v) for v in box_xyxy]
    if max(b) <= 1.5:                         # normalized cx,cy,w,h
        cx, cy, w, h = b
        x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
    else:                                     # pixel x1,y1,x2,y2
        x1, y1, x2, y2 = b
    x1 = int(round(min(x1, x2))); x2 = int(round(max(x1, x2)))
    y1 = int(round(min(y1, y2))); y2 = int(round(max(y1, y2)))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def m_glare_from_burn(burn_mask, box_xyxy, torso_weight=True, head_top_weight=0.4):
    """``m_glare`` = torso-weighted fraction of the body box that is NOT burnt.

    Args:
        burn_mask:  HxW array; nonzero where pixels are photometrically destroyed
                    (saturated / burnt). Anything >0 counts as burnt.
        box_xyxy:   person box, normalized (cx,cy,w,h) or pixel (x1,y1,x2,y2).
        torso_weight: down-weight the top rows (head) so a burnt head/lamp barely
                    lowers m_glare (spec v4 §4: "mất đầu vì đèn = Band A").
        head_top_weight: weight at the very top row (linearly rises to 1.0 at bottom).

    Returns float in [0,1]. 1.0 = body fully intact; ->0 = body burnt out.
    """
    M = np.asarray(burn_mask)
    H, W = M.shape[:2]
    x1, y1, x2, y2 = _box_to_px(box_xyxy, W, H)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    burnt = (M[y1:y2, x1:x2] > 0).astype(np.float32)
    intact = 1.0 - burnt
    if torso_weight:
        wy = np.linspace(head_top_weight, 1.0, intact.shape[0], dtype=np.float32)[:, None]
        ww = np.broadcast_to(wy, intact.shape)
        m = float((intact * ww).sum() / (ww.sum() + 1e-6))
    else:
        m = float(intact.mean())
    return float(np.clip(m, 0.0, 1.0))


def combine_m(m_glare, vis_geom):
    """Unified reliability ``m = m_glare * vis_geom`` (spec v4 §4)."""
    return float(np.clip(m_glare, 0.0, 1.0) * np.clip(vis_geom, 0.0, 1.0))


def classify_provenance(m_glare, vis_geom, tau=PROV_TAU):
    """Offline provenance label from the two axes (spec v4 §3).

    Runtime adds 'predicted' (self-forecast); offline GT only ever sees the four
    observable causes below.
    """
    glare = m_glare < tau
    occ = vis_geom < tau
    if glare and occ:
        return "occ+glare"
    if glare:
        return "glare"
    if occ:
        return "occluded"
    return "detected"


def visibility_band(m):
    """Map unified ``m`` to a (band, name, mode) per the spec v4 §4 4-band ladder."""
    m = float(np.clip(m, 0.0, 1.0))
    for lo, band, name, mode in BAND_THRESHOLDS:
        if m >= lo:
            return band, name, mode
    return "D", "LOST", "FORECAST"           # unreachable (lo=-1 catches all)


def make_visibility_label(box_xyxy, burn_mask, vis_geom=1.0, foot_y=None,
                          beacon_present=False, tau=PROV_TAU, torso_weight=True):
    """Full offline label for one (agent, frame): the §13 extra columns.

    Returns a dict with ``m_glare, vis_geom, m, prov, band, mode, foot_y,
    beacon_present``. ``vis_geom`` is expected from :func:`occlusion_graph` (or 1.0
    when there is a single agent / no occlusion).
    """
    m_glare = m_glare_from_burn(burn_mask, box_xyxy, torso_weight=torso_weight)
    vis_geom = float(np.clip(vis_geom, 0.0, 1.0))
    m = combine_m(m_glare, vis_geom)
    prov = classify_provenance(m_glare, vis_geom, tau=tau)
    band, name, mode = visibility_band(m)
    return {
        "m_glare": m_glare, "vis_geom": vis_geom, "m": m,
        "prov": prov, "band": band, "band_name": name, "mode": mode,
        "foot_y": (None if foot_y is None else float(foot_y)),
        "beacon_present": bool(beacon_present),
    }
