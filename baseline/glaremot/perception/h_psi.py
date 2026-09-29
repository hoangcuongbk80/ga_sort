"""Learned head h_psi (hybrid Stage-1) - m_glare on a frozen YOLO backbone tap.

Hybrid split: the ANALYTIC block (`analytic.py`) owns the lamp core, the
beacon point ``mu``, and ``m_glare_analytic``. This learned head adds ONLY the
semantics rules can't reach, on the FROZEN detector backbone (~free):

  - m_glare in [0,1] : localization reliability; REFINES ``m_glare_analytic``
    (its regression target) with image context. This is the sole head output, and
    the only learned signal the controller consumes at inference. Lamp-rejection at
    birth is analytic (the saturated-core fraction ``core_frac``), and the beacon
    point is the analytic core centroid, so the head needs no presence/Sigma output.

POOL-FIRST design: because the backbone is frozen, the roi_align happens on
the frozen feature map and the trainable head operates on the pooled
[N, C, roi, roi] tensor. This lets us PRECOMPUTE the pooled features ONCE
(scripts/05) and then train the tiny head for many epochs straight from
RAM/GPU - no per-epoch image I/O or backbone forward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

__all__ = ["YoloBackboneTap", "HybridVisibilityHead", "HPsiModel",
           "Stage1HeadLoss", "boxes_to_rois"]


class YoloBackboneTap(nn.Module):
    """Partial forward of an ultralytics DetectionModel up to a neck layer
    (default: P3, stride 8). Weights frozen; channels/stride are inferred with
    a dummy forward so any YOLO variant (n..x) works unchanged."""

    def __init__(self, detection_model, tap_index=16, freeze=True):
        super().__init__()
        self.seq = detection_model.model          # nn.Sequential (each layer has .f)
        self.tap_index = tap_index
        if freeze:
            for p in self.seq.parameters():
                p.requires_grad_(False)
        self.eval()
        with torch.no_grad():
            feat = self._partial_forward(torch.zeros(1, 3, 64, 64))
        self.out_channels = feat.shape[1]
        self.stride = 64 // feat.shape[-1]

    def _partial_forward(self, x):
        y = []
        for i, m in enumerate(self.seq):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if i == self.tap_index:
                return x
        return x

    def forward(self, img):
        with torch.no_grad():                     # backbone is frozen, always
            return self._partial_forward(img)


def boxes_to_rois(boxes_xyxy, batch_idx):
    """rois [N,5] = (batch_idx, x1, y1, x2, y2) from pixel boxes [N,4]."""
    t = torch.as_tensor(boxes_xyxy, dtype=torch.float32).view(-1, 4)
    bidx = torch.as_tensor(batch_idx, dtype=torch.float32).view(-1, 1)
    return torch.cat([bidx, t], dim=1)


class HybridVisibilityHead(nn.Module):
    """Trainable head on POOLED frozen features [N,C,roi,roi]."""

    def __init__(self, in_channels, roi_size=7):
        super().__init__()
        self.roi_size = roi_size
        c = 64
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(c * roi_size * roi_size, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )
        self.m_head = nn.Linear(128, 1)

    def forward(self, pooled):
        z = self.mlp(self.roi_conv(pooled).flatten(1))
        return {
            "m_glare": torch.sigmoid(self.m_head(z)).squeeze(-1),
        }


class HPsiModel(nn.Module):
    """Frozen YOLO backbone tap + trainable hybrid visibility head."""

    def __init__(self, detection_model, tap_index=16, roi_size=7):
        super().__init__()
        self.backbone = YoloBackboneTap(detection_model, tap_index=tap_index, freeze=True)
        self.head = HybridVisibilityHead(self.backbone.out_channels, roi_size)
        self.stride = self.backbone.stride

    @torch.no_grad()
    def pool(self, img, rois):
        """Frozen ROI features [N,C,roi,roi] for rois [N,5] (image px). Cacheable."""
        feat = self.backbone(img)
        return roi_align(feat, rois, output_size=self.head.roi_size,
                         spatial_scale=1.0 / self.stride, aligned=True)

    def forward(self, img, rois):
        return self.head(self.pool(img, rois))


class Stage1HeadLoss(nn.Module):
    """L = lambda_m * m (person ROIs). ``lambda_present``/``lambda_sigma`` are
    accepted but ignored: the head carries only the observability output ``m_glare``
    at inference (lamp-rejection is analytic, via ``core_frac``; the beacon point is
    the analytic core centroid), so there is no presence or Sigma output to supervise."""

    def __init__(self, lambda_present=0.0, lambda_m=1.0, lambda_sigma=0.0,
                 focal_gamma=2.0, focal_alpha=0.75):
        super().__init__()
        self.lm = lambda_m

    def forward(self, pred, target):
        """target keys: m_glare_analytic [N]; m_valid [N] bool (person ROIs)."""
        mvalid = target.get("m_valid")
        if mvalid is not None and mvalid.any():
            l_m = F.smooth_l1_loss(pred["m_glare"][mvalid], target["m_glare_analytic"][mvalid])
        else:
            l_m = pred["m_glare"].sum() * 0.0
        return {"total": self.lm * l_m, "L_m": l_m}
