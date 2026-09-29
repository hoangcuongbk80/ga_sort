"""Analytic perception block - rule-based, NO training (HBB version).

The hybrid Stage-1 splits perception in two: this analytic block owns
"detection of light" - the lamp core, the beacon point ``mu``, and the
photometric reliability ``m_glare`` - computed with fixed, physics-based rules
that are identical on real and synthetic frames (domain-invariant). The
LEARNED head ``h_psi`` only adds an image-context ``m_glare`` refinement (its
sole inference output); ``m_glare_analytic`` is the target it regresses to.

Crucially, ``m_glare_analytic`` is the TARGET the head regresses to, and it is
computed FROM THE IMAGE (never from the synthetic engine), so the head never
learns a synthetic artifact.

Boxes are axis-aligned: people stand upright, so the torso axis is vertical
(head = top of the box, feet = bottom). All inputs are pixel-space:
``bgr`` HxWx3 uint8 and ``box_px = (x1, y1, x2, y2)``.
"""
import numpy as np

from ..photometric import saturation_map, saturated_core_mask

__all__ = [
    "m_glare_analytic", "head_roi_box",
    "beacon_mu_analytic", "beacon_present_analytic", "analytic_perception",
]
_EPS = 1e-6


def _clip_box(x1, y1, x2, y2, H, W):
    x1 = int(max(0, min(W - 1, round(x1))))
    x2 = int(max(0, min(W, round(x2))))
    y1 = int(max(0, min(H - 1, round(y1))))
    y2 = int(max(0, min(H, round(y2))))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def m_glare_analytic(bgr, box_px, head_weight=0.4, v_thr=0.92, s_max=0.35):
    """Torso-weighted fraction of the body box that is NOT photometrically burnt.

    1.0 = body intact; -> 0 = burnt out. The weight grows linearly from
    ``head_weight`` at the top edge (the lamp sits on the helmet - a burnt head
    barely lowers m) to 1.0 at the feet (a burnt torso/legs is real feature
    loss). Degenerate/empty box -> 1.0 (no evidence of burn).
    """
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(*box_px, H, W)
    sat = saturation_map(bgr[y1:y2, x1:x2], v_thr=v_thr, s_max=s_max)
    if sat.size == 0:
        return 1.0
    intact = (~sat).astype(np.float64)                       # 1 where pixel survives
    h = sat.shape[0]
    dist = (np.arange(h, dtype=np.float64) + 0.5) / max(h, 1)  # 0 head .. 1 feet
    weight = head_weight + (1.0 - head_weight) * dist          # [h]
    w_col = np.repeat(weight[:, None], sat.shape[1], axis=1)
    m = float((w_col * intact).sum() / (w_col.sum() + _EPS))
    return float(np.clip(m, 0.0, 1.0))


def head_roi_box(box_px, head_band=0.5, expand_frac=0.15):
    """Head ROI: the top ``head_band`` fraction of the box, expanded
    ``expand_frac`` of the box height ABOVE the top edge (the lamp pokes above
    the head). Returns (x1, y1, x2, y2) floats."""
    x1, y1, x2, y2 = [float(v) for v in box_px]
    bh = max(y2 - y1, _EPS)
    return x1, y1 - expand_frac * bh, x2, y1 + head_band * bh


def beacon_mu_analytic(core_mask, box_px, head_band=0.5, expand_frac=0.15):
    """Analytic beacon point mu: area-centroid of the compact lamp core inside
    the head ROI. Returns (mu [x, y] float32 | None, core_pixel_count)."""
    core = np.asarray(core_mask)
    H, W = core.shape[:2]
    rx1, ry1, rx2, ry2 = _clip_box(*head_roi_box(box_px, head_band, expand_frac), H, W)
    sub = core[ry1:ry2, rx1:rx2]
    ys, xs = np.nonzero(sub)
    if xs.size == 0:
        return None, 0
    mu = np.array([xs.mean() + rx1, ys.mean() + ry1], np.float32)
    return mu, int(xs.size)


def beacon_present_analytic(core_mask, box_px, head_band=0.5, expand_frac=0.15,
                            min_core_px=8):
    """Geometric present gate: a lamp core exists in this person's head band.
    Returns (present_bool, mu | None, core_pixel_count)."""
    mu, count = beacon_mu_analytic(core_mask, box_px, head_band, expand_frac)
    return (count >= min_core_px and mu is not None), mu, count


def analytic_perception(bgr, box_px, core_mask=None, **kw):
    """Run the whole analytic block for one box.

    Returns dict {m_glare, mu, present, core_px}. This is the per-(person,
    frame) analytic feature used (a) as target + context for the learned head
    ``h_psi`` and (b) directly by the controller as the photometric fallback.
    """
    if core_mask is None:
        core_mask = saturated_core_mask(bgr)
    m_glare = m_glare_analytic(
        bgr, box_px, head_weight=kw.get("head_weight", 0.4),
        v_thr=kw.get("v_thr", 0.92), s_max=kw.get("s_max", 0.35))
    present, mu, count = beacon_present_analytic(
        core_mask, box_px, head_band=kw.get("head_band", 0.5),
        expand_frac=kw.get("expand_frac", 0.15), min_core_px=kw.get("min_core_px", 8))
    return {"m_glare": m_glare, "mu": mu, "present": present, "core_px": count}
