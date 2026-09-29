r"""GlareTrack - glare-robust, motion-only, appearance-free multi-object tracker.

The official tracker for the GlareMOT benchmark (mining-glare MOT). Motion backbone
is OC-SORT (Kalman prediction, velocity-direction association, OCR rematching;
vendored in ``glaremot.external.ocsort``). Appearance/Re-ID is NOT used: everyone
wears identical PPE, and our ablation showed appearance embeddings add nothing once
the learned motion prior is in place.

Motion is pure Kalman constant-velocity (OC-SORT). There is no learned forecaster:
an ablation on the final stack showed the IMLE rollout / DiffMOT-style association
prior was redundant once the glare controller (overlap-band birth + tight output)
and the lamp beacon were in place - removing it was neutral-to-better on the
real test and left the held recovery unchanged.

The glare controller - "glare changes the rules at every layer" (the lamp must never
become a person, the burnt person must not be lost):
  - DETECTION: each detection carries a photometric ``core_frac`` (saturated
    fraction) and the learned observability ``m_glare`` from the head ``h_psi``;
    near-pure light blobs (high ``core_frac``) can neither associate nor birth, so
    lamp-rejection is the analytic ``core_frac`` gate.
  - BIRTH (overlap-band suppression): a candidate is suppressed only as a DUPLICATE
    (high IoU with an existing track), never merely for being center-close to a
    distinct neighbour - so two people walking together are both tracked.
  - FUSION: ``m`` scales the Kalman measurement noise R by 1/m (a weight, not a gate);
    the fused state goes to the box history (velocity clamp) with an honest m.
  - SURVIVAL: while a confirmed track is unobserved, the analytic lamp core inside
    the predicted head band acts as a weak center-only BEACON; the Kalman coast + an
    earned fill budget keep the box through a burn; a re-entry gate guards revival.
  - LOCALIZATION: an observed track emits its tight raw detection box; only a coasting
    track emits the smoothed prediction.
  - IDENTITY: contested detections are re-awarded by last observation; order-inverting
    pair assignments are swapped back.
  - EXIT: an unobserved track whose last motion carried it out through a frame edge
    dies fast and loses every contested detection.

Boxes are axis-aligned (cx, cy, w, h), normalized to the frame.
"""
import argparse
import csv
import glob
import os
import queue
import random
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Ultralytics"))

from ..config import data_path, load_config, results_dir, weight_path
from ..data.splits import make_splits
from ..external.ocsort.association import associate as oc_associate
from ..external.ocsort.association import iou_batch, linear_assignment
from ..external.ocsort.ocsort import KalmanBoxTracker, k_previous_obs
from ..perception.h_psi import HybridVisibilityHead
from ..photometric import core_centroid_in_box, mask_area_inside
from .occlusion_graph import compute_vis_geom

# --- reliability / hysteresis ---
TAU_HI = 0.45            # m_glare >= TAU_HI: reliable evidence (grows good_streak)
TAU_LO = 0.25            # m_glare <  TAU_LO: not person evidence at all
TRACK_THRESH = 0.20
RELIABLE_DWELL = 2       # consecutive good frames before flipping back to RELIABLE

# --- birth / confirmation (the lamp must not become a person) ---
BIRTH_CONF = float(os.environ.get("GMOT_BIRTH_CONF", "0.35"))
# Deployment-only flare birth-guard: veto a BRAND-NEW track while the whole frame is
# blooming (global saturated-pixel fraction > threshold). A flare washes out real
# people (their tracks coast) and the detector fires on the bloom itself; a fresh
# detection appearing mid-flare is almost always that artifact, not a new person.
# Normal frames sit at ~0.007 saturated; a flare spikes to 0.2-0.3. Baked default 0.05
# vetoes those mid-flare phantom births (free on the benchmark, +0.02 HOTA / fewer FP);
# set 0.0 to disable. Only fires when step() is passed the frame's flare fraction.
FLARE_BIRTH_GUARD = float(os.environ.get("GMOT_FLARE_BIRTH_GUARD", "0.05"))
# Frames to keep vetoing births AFTER the flare drops below threshold: the bloom
# lingers a frame or two and the detector fires on the fading glow as it clears.
FLARE_BIRTH_COOL = int(os.environ.get("GMOT_FLARE_BIRTH_COOL", "8"))
BIRTH_M_MIN = 0.40       # torso-weighted m_glare floor for a brand-new track
BIRTH_CORE_MAX = 0.40    # box mostly saturated core -> lamp, never birth
ASSOC_CORE_MAX = 0.85    # near-pure light blob -> not even a measurement
BIRTH_SUPPRESS_IOU = float(os.environ.get("GMOT_BIRTH_SUPPRESS_IOU", "0.35"))
# Center-distance birth suppression. ROOT CAUSE of the big suppressed-FN seqs
# (211/176/...): two DISTINCT people walking close (centers within 0.55·scale but
# boxes not overlapping) - the second is blocked from birth for the whole co-presence
# even with strong detections, until the first leaves. TrackTrack's track-aware NMS
# suppresses by IoU ONLY (no center term), which is why it tracks these scenes. Set
# to 0 to disable the center test and rely on IoU suppression alone. Env-tunable.
BIRTH_SUPPRESS_CENTER_FRAC = float(os.environ.get("GMOT_BIRTH_SUPPRESS_CENTER", "0.55"))
# Overlap-band birth suppression (NOVEL). The center-distance test above is only
# meant to kill a duplicate (one person, two detector boxes, partial overlap); applied
# unconditionally it ALSO blocks a DISTINCT person walking close with little overlap
# (e.g. seq 211/305: a 2nd person blocked from birth for hundreds of frames despite a
# strong detection). So the center test fires ONLY inside the duplicate overlap band,
# ref_iou >= BIRTH_CENTER_MIN_IOU; a more separated candidate is treated as a new
# object and always allowed to birth. This is the appearance-free, track-aware spirit
# of TrackTrack's NMS (suppress by IoU, not by a blind center radius).
BIRTH_CENTER_MIN_IOU = float(os.environ.get("GMOT_BIRTH_CENTER_MIN_IOU", "0.20"))
# Ablation toggle (mechanism 2, track-aware birth): default 1 = on. Set 0 to drop the
# track-aware overlap-band/center suppression entirely (a candidate may birth regardless
# of existing tracks), for the "- overlap birth" ablation row. Does not change defaults.
OVERLAP_BIRTH = os.environ.get("GMOT_OVERLAP_BIRTH", "1") not in ("0", "off", "false", "")
# consecutive reliable frames before a track is real. The lamp-flicker defence,
# but on hard NON-glare sequences (dark, crouching, sporadically detected) a
# person rarely gets 6 consecutive good frames -> never births -> coasting never
# engages -> the whole track is FN (seq 128: 0.52^6 ~ 2% chance to confirm).
# Env-tunable so the recall/precision trade can be swept.
CONFIRM_HITS = int(os.environ.get("GMOT_CONFIRM_HITS", "2"))
TENTATIVE_MAX_MISS = 2

TENTATIVE_MAX_AGE = 30

# --- association / motion ---
IOU_THRESH = 0.30
INERTIA = 0.20
DELTA_T = 3
REENTRY_GAP = 8          # gap above which a match must pass the re-entry gate
REENTRY_M_MIN = 0.35

# --- fusion weights ---
R_M_FLOOR = 0.15         # R scale = 1 / max(m, R_M_FLOOR)
HIST_M_MIN = 1e-3

# --- coast / fill budget ---
FORECAST_START = 10      # gap past which Kalman velocity is damped (long unobserved)
HIST_LEN = 8             # box-trajectory window kept for the velocity-step clamp
MAX_AGE = 160            # hard cap (covers the longest observed glare gap)
FILL_BASE = 12
FILL_PER_HIT = 6         # earned hold: gap allowed = FILL_BASE + FILL_PER_HIT*good_hits
GAMMA = 0.985
EMA = float(os.environ.get("GMOT_EMA", "0.5"))   # coasting-box smoothing; lower = tighter localization
# When observed (gap==0), emit the RAW detection box (tight, best GT-IoU) instead of
# the EMA-smoothed track box (laggy/loose) - recovers loose-box-on-person near-misses
# that the strict IoU>0.5 criterion double-penalizes. Coasting frames still emit smooth.
TIGHT_OUTPUT = os.environ.get("GMOT_TIGHT_OUTPUT", "1") not in ("0", "off", "false", "")
FC_MIN_GOOD_HITS = 20    # a young track has no trustworthy velocity to coast on
# A confirmed track is kept ALIVE up to allowed_gap (for re-association through a
# burn, so its ID survives), but it is only EMITTED as a box while it has been
# observed within OUTPUT_MAX_GAP frames. Holding a box on pure prediction far past
# the last observation is the dominant false-positive source on a strong detector
# (the person has usually just left / been briefly missed, not burnt). Env-tunable.
OUTPUT_MAX_GAP = int(os.environ.get("GMOT_OUTPUT_MAX_GAP", "12"))   # OFFICIAL deployed cap G=12
# A BEACON hold is NOT pure prediction: the lamp core re-localises the box every
# frame, so it stays on the person through a flare instead of drifting like Kalman
# coast. The emission cap is a SINGLE knob G: the pure-Kalman coast and the
# beacon-anchored coast share the same cap, so the two env vars are kept coupled
# (set both to change G). The OFFICIAL deployed operating point is G=12, which
# reproduces the paper's real-test HOTA 69.17 (DetA 65.40 / AssA 74.10 / IDF1 85.19
# / IDSW 14 / FP 1710) out of the box; the recovery benchmark raises only this cap
# to G=80 (see eval/eval_recovery.py --max_gap, which couples both vars).
OUTPUT_MAX_GAP_BEACON = int(os.environ.get("GMOT_OUTPUT_MAX_GAP_BEACON", "12"))
VEL_DAMP = 0.95          # per-frame Kalman velocity decay while unobserved past FORECAST_START

# --- exit gate (a person who walked OUT of the frame must die fast) ---
# The detector loses a leaving person BEFORE the box touches the edge (half a
# body left), so the test is projective: at the observed outward speed, would
# the person cross the edge within EXIT_HORIZON frames? Floored/capped so a
# glare-occluded person mid-frame can never qualify.
EXIT_BORDER_FRAC = 0.04
EXIT_MARGIN_MAX = 0.12
EXIT_HORIZON = 20
EXIT_VEL_MIN = 8e-4
EXIT_MAX_GAP = 8
EXIT_REASSOC_GAP = 3

# --- BYTE-lite: low-score detections may SUSTAIN a recent track, never create one ---
BYTE_M_MIN = 0.12
BYTE_CORE_MAX = 0.60
BYTE_IOU = 0.25

# --- identity protection in close groups (anti ID-steal/swap, motion-only) ---
AMBIG_IOU = 0.20
AMBIG_MARGIN = 0.10
SWAP_SEP_FRAC = 0.25

# --- photometric runtime (uint8 / downscaled twins of photometric.py) ---
PHOTO_S = 512
SAT_V_MIN = 235          # = 0.92 * 255
SAT_S_MAX = 89           # = 0.35 * 255
CORE_MIN_AREA = 4
CORE_FILL_MIN = 0.55
CORE_ECC_MAX = 0.85

# --- fixed-camera scene prior (online static-background model, default OFF) ---
# Mining cameras are fixed, so static structures + fixed lights + floor reflections
# are temporally constant while real workers (headlamp flicker + micro-motion in the
# dark drift) always produce foreground residual (audit: 0% of GT worker-frames are
# background-static). A per-sequence running background separates a high-conf detector
# HALLUCINATION on a static person-shaped structure (low residual -> veto birth) from a
# genuine worker (high residual). Default off => det dicts byte-identical.
SCENE_ENABLE = os.environ.get("GMOT_SCENE", "1") not in ("0", "off", "false", "")   # OFFICIAL baked ON
SCENE_ALPHA = float(os.environ.get("GMOT_SCENE_ALPHA", "0.02"))   # EMA bg rate (~1/alpha frame memory)
SCENE_WARMUP = int(os.environ.get("GMOT_SCENE_WARMUP", "8"))      # frames before the residual is trusted
SCENE_VETO = os.environ.get("GMOT_SCENE_VETO", "1") not in ("0", "off", "false", "")   # OFFICIAL baked ON
SCENE_VETO_THR = float(os.environ.get("GMOT_SCENE_VETO_THR", "1.5"))  # birth needs residual > THR*frame-median
# --- beacon (weak center-only pseudo-measurement from the lamp core) ---
BEACON_MIN_PX = 2
BEACON_HEAD_BAND = 0.45
BEACON_EXPAND_UP = 0.15
BEACON_GATE_FRAC = 0.75


def xyxy_to_norm_box(xyxy, W, H):
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    x1 = max(0.0, min(float(W - 1), x1)); x2 = max(0.0, min(float(W - 1), x2))
    y1 = max(0.0, min(float(H - 1), y1)); y2 = max(0.0, min(float(H - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    w = max(1.0, x2 - x1) / float(W)
    h = max(1.0, y2 - y1) / float(H)
    return np.array([((x1 + x2) * 0.5) / W, ((y1 + y2) * 0.5) / H, w, h], np.float32)


def norm_box_to_xyxy(box, W, H):
    cx, cy, w, h = [float(v) for v in box[:4]]
    return np.array([cx * W - w * W * 0.5, cy * H - h * H * 0.5,
                     cx * W + w * W * 0.5, cy * H + h * H * 0.5], np.float32)


def reliability_color(m):
    m = float(np.clip(m, 0.0, 1.0))
    return (0, int(255 * m), int(255 * (1.0 - m)))


def smooth_box(prev, target):
    if prev is None:
        return target.copy()
    return (EMA * prev + (1.0 - EMA) * target).astype(np.float32)


def fast_saturation_map(bgr):
    """uint8 twin of photometric.saturation_map (same thresholds, no float pass)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] >= SAT_V_MIN) & (hsv[..., 1] <= SAT_S_MAX)


def fast_core_mask(sat_bool):
    """Compact lamp-core mask from a precomputed saturation map; same component
    filtering as photometric.saturated_core_mask with PHOTO_S-scaled min_area."""
    from ..photometric import _eccentricity
    core = sat_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    out = np.zeros_like(core)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < CORE_MIN_AREA:
            continue
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area / float(max(1, w * h)) < CORE_FILL_MIN:
            continue
        comp = labels == i
        if _eccentricity(comp) >= CORE_ECC_MAX:
            continue
        out[comp] = 1
    return out


class Perception:
    """YOLO detector + frozen-backbone h_psi head + photometric runtime.

    One detector forward serves detection AND the h_psi head: a forward hook on
    the tap layer captures the P3 feature map, which is roi_aligned per box.
    """

    def __init__(self, best, head_ckpt, device, conf):
        os.environ.setdefault("YOLO_CONFIG_DIR", os.path.dirname(os.path.abspath(best)))
        from ultralytics import YOLO
        self.detector = YOLO(best)
        self.device = device
        self.conf = conf
        ck = torch.load(head_ckpt, weights_only=False)
        meta = ck["meta"]
        self.S = int(meta["imgsz"])
        self.stride = meta["stride"]
        self.tap = meta["tap"]
        self.head = HybridVisibilityHead(meta["in_channels"],
                                         roi_size=meta["roi_size"]).to(device).eval()
        # present_head/sigma_head were training-time auxiliaries; they are removed
        # from the deployed head, so drop their checkpoint weights and load the
        # observability (m_glare) head alone.
        sd = {k: v for k, v in ck["state_dict"].items()
              if not k.startswith("present_head") and not k.startswith("sigma_head")}
        self.head.load_state_dict(sd)
        self._feats = {}
        self._sat = None
        self._core = None
        self._scene_bg = None      # running grayscale background (PHOTO_S x PHOTO_S)
        self._scene_n = 0          # frames accumulated into the background
        self.detector.model.model[self.tap].register_forward_hook(
            lambda _m, _i, o: self._feats.__setitem__("p3", o))

    def reset_scene(self):
        """Drop the static-background model at a sequence boundary (Perception is
        shared across sequences; the scene prior is per fixed-camera video)."""
        self._scene_bg = None
        self._scene_n = 0

    def _scene_update(self, small_bgr):
        """Update the EMA background from the current PHOTO_S frame and return the
        per-pixel absolute residual map + a robust frame-level residual scale."""
        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._scene_bg is None:
            self._scene_bg = gray.copy()
            self._scene_n = 1
            return None, None
        resid = np.abs(gray - self._scene_bg)
        # update AFTER measuring residual so a worker doesn't suppress their own signal
        self._scene_bg += SCENE_ALPHA * (gray - self._scene_bg)
        self._scene_n += 1
        scale = float(np.median(resid)) + 1.0
        return resid, scale

    def core_mask(self):
        """Compact lamp-core mask of the CURRENT frame (PHOTO_S x PHOTO_S),
        computed lazily and cached until the next __call__."""
        if self._core is None and self._sat is not None:
            self._core = fast_core_mask(self._sat)
        return self._core, PHOTO_S

    @torch.no_grad()
    def __call__(self, frame):
        S = self.S
        H0, W0 = frame.shape[:2]
        square = (H0 == S and W0 == S)
        # Detector input: aspect-preserving LETTERBOX, not a distorting square
        # resize. On already-square frames (the 1536^2 GlareMOT benchmark) this is
        # the identity, so benchmark behaviour is byte-identical; on real non-square
        # video it removes the horizontal/vertical squash that otherwise collapses
        # the detector (12% -> 97% recall on a 16:9 mine clip).
        if square:
            det_img = frame
            r, x0, y0 = 1.0, 0, 0
        else:
            r = min(S / float(W0), S / float(H0))
            nw, nh = int(round(W0 * r)), int(round(H0 * r))
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            det_img = np.full((S, S, 3), 114, dtype=frame.dtype)  # YOLO pad grey
            x0, y0 = (S - nw) // 2, (S - nh) // 2
            det_img[y0:y0 + nh, x0:x0 + nw] = resized
        # Photometric / scene maps keep the original-frame square convention
        # (sourced from `frame`, never the padded det_img) so letterbox padding can
        # never spawn phantom saturation cores or scene residual. Deployment on a
        # non-square video should pre-letterbox the clip to S x S (the same square the
        # MOT benchmark uses) before tracking.
        small = cv2.resize(frame, (PHOTO_S, PHOTO_S), interpolation=cv2.INTER_AREA)
        self._sat = fast_saturation_map(small)
        self.flare_frac = float(self._sat.mean())   # global bloom level (deployment guard)
        self._core = None
        scene_resid, scene_scale = (None, None)
        if SCENE_ENABLE:
            scene_resid, scene_scale = self._scene_update(small)
        res = self.detector.predict(det_img, imgsz=S, conf=self.conf, verbose=False,
                                    device=self.device)[0]
        dets = []
        if res.boxes is None or not len(res.boxes):
            return dets

        sat = self._sat
        ps = PHOTO_S / float(S)
        # boxes in LETTERBOX-S coords drive roi_align (the hooked P3 map came from
        # det_img); a copy mapped back into the original square-resize-S convention
        # drives every downstream geometry consumer, so the rest of __call__ and the
        # whole tracker are unchanged. On square frames the two are identical.
        xyxy_lb = res.boxes.xyxy.detach()
        confs = res.boxes.conf.detach().cpu().numpy()
        rois = torch.cat([torch.zeros(len(xyxy_lb), 1, device=xyxy_lb.device), xyxy_lb], dim=1)
        pooled = roi_align(self._feats["p3"].float(), rois,
                           output_size=self.head.roi_size,
                           spatial_scale=1.0 / self.stride, aligned=True)
        out = self.head(pooled)
        # The head outputs only the learned observability ``m_glare``; lamp-rejection
        # at birth is handled by the analytic saturated-core gate ``core_frac``.
        mg = out["m_glare"].cpu().numpy()

        xyxy = xyxy_lb.cpu().numpy().copy()
        if not square:
            # undo letterbox (-> original frame px) then apply the legacy square
            # resize (-> S-square px) so downstream S-normalisation is preserved.
            # Use the INTEGER insertion offset (x0, y0) the image actually sits at,
            # not the float pad, so odd padding stays sub-pixel exact.
            xyxy[:, 0] = (xyxy[:, 0] - x0) / r * (S / float(W0))
            xyxy[:, 2] = (xyxy[:, 2] - x0) / r * (S / float(W0))
            xyxy[:, 1] = (xyxy[:, 1] - y0) / r * (S / float(H0))
            xyxy[:, 3] = (xyxy[:, 3] - y0) / r * (S / float(H0))
            xyxy = np.clip(xyxy, 0.0, float(S))   # whole-array clip (writes back)

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            ix1, iy1 = max(0, int(x1 * ps)), max(0, int(y1 * ps))
            ix2, iy2 = min(PHOTO_S, int(x2 * ps) + 1), min(PHOTO_S, int(y2 * ps) + 1)
            patch = sat[iy1:iy2, ix1:ix2]
            core_frac = float(patch.mean()) if patch.size else 0.0
            box = np.array([((x1 + x2) * 0.5) / S, ((y1 + y2) * 0.5) / S,
                            max(1.0, x2 - x1) / S, max(1.0, y2 - y1) / S], np.float32)
            det = {
                "box": box,
                "xyxy_s": np.array([x1, y1, x2, y2], np.float32),
                "conf": float(confs[i]),
                "m_glare": float(np.clip(mg[i], 0.0, 1.0)),
                "core_frac": core_frac,
            }
            if SCENE_ENABLE:
                # relative foreground residual inside the box: >>1 = moving worker,
                # ~1 = static structure. inf during warmup so it never vetoes early.
                if scene_resid is None or self._scene_n <= SCENE_WARMUP:
                    det["bg_resid"] = float("inf")
                else:
                    rb = scene_resid[iy1:iy2, ix1:ix2]
                    det["bg_resid"] = (float(np.median(rb)) / scene_scale
                                       if rb.size else float("inf"))
            dets.append(det)
        return dets


class MotionTrack:
    """OC-SORT Kalman track + glare-controller state.

    States: TENTATIVE (not yet a person) -> RELIABLE / DEGRADED (observed,
    high/low m) -> LOST (unobserved: KALMAN, BEACON or FORECAST fill). Trust is
    gained slowly and lost instantly (one bad frame -> DEGRADED)."""

    def __init__(self, kf, det, W, H, frame_idx):
        self.kf = kf
        self.R0 = kf.kf.R.copy()
        self.id = int(kf.id + 1)
        self.hist = deque(maxlen=HIST_LEN)
        self.centers = deque(maxlen=96)
        self.exiting = False     # last observed motion was carrying the person out
        self.state = "TENTATIVE"
        self.mode = "TENTATIVE"
        self.is_confirmed = False
        self.good_streak = 0
        self.good_hits = 0
        self.age_frames = 0
        self.beacon = None
        self.m_glare = float(det["m_glare"])
        self.vis_geom = 1.0
        self.m = 0.0
        self.last_frame = frame_idx
        self.box = xyxy_to_norm_box(kf.get_state()[0], W, H)
        self.smooth = self.box.copy()
        self.det_box = np.array(det["box"], np.float32)  # raw detector box (tight) for output
        self._push_hist(self.box, max(self.m_glare, HIST_M_MIN))

    def out_box(self):
        """Box to EMIT: the tight raw detection box when observed this frame, else
        the smoothed track box (coasting). Internal association always uses smooth."""
        if TIGHT_OUTPUT and int(self.kf.time_since_update) == 0 and self.det_box is not None:
            return self.det_box
        return self.smooth

    def earned_gap(self):
        """OC-SORT earned-coast budget: the keep-alive gap governed purely by hit
        history and the exit gate."""
        if not self.is_confirmed:
            return TENTATIVE_MAX_MISS
        g = min(MAX_AGE, FILL_BASE + FILL_PER_HIT * self.good_hits)
        if self.exiting:
            g = min(g, EXIT_MAX_GAP)   # the person left the scene: no earned hold applies
        return g

    def allowed_gap(self):
        return self.earned_gap()

    def update_exit_state(self, W, H):
        """An UNOBSERVED track whose last observation sat near a frame edge
        while its observed motion pointed out through that same edge: the
        person walked out of view. Glare/occlusion gaps happen mid-frame and
        never satisfy the border+outward test; a person merely standing at the
        doorway keeps near-zero velocity -> not exiting."""
        if self.exiting:
            return True
        lo = self.kf.last_observation[:4]
        if lo[0] < 0 or len(self.centers) < 3:
            return False
        c = np.asarray(list(self.centers)[-5:], np.float32)
        steps = max(len(c) - 1, 1)
        vx = float(c[-1, 0] - c[0, 0]) / steps   # normalized units/frame
        vy = float(c[-1, 1] - c[0, 1]) / steps

        def reach(v, dim):
            return min(max(EXIT_BORDER_FRAC * dim, abs(v) * dim * EXIT_HORIZON),
                       EXIT_MARGIN_MAX * dim)

        self.exiting = bool(
            (vx < -EXIT_VEL_MIN and lo[0] <= reach(vx, W))
            or (vx > EXIT_VEL_MIN and W - lo[2] <= reach(vx, W))
            or (vy < -EXIT_VEL_MIN and lo[1] <= reach(vy, H))
            or (vy > EXIT_VEL_MIN and H - lo[3] <= reach(vy, H))
        )
        return self.exiting

    def median_step_px(self, W, H):
        hist = np.asarray(self.hist, np.float32)
        if len(hist) < 3:
            return 0.004 * max(W, H)
        steps = np.hypot(np.diff(hist[:, 0]) * W, np.diff(hist[:, 1]) * H)
        return float(np.median(steps))

    def _push_hist(self, box, m):
        row = np.array([box[0], box[1], box[2], box[3], m], np.float32)
        self.hist.append(row)
        self.centers.append((float(box[0]), float(box[1])))

    def update_detection(self, det, W, H, frame_idx, gap_before):
        self.beacon = None
        self.exiting = False  # re-observed: the person is (still) in the frame
        m_det = float(det["m_glare"])
        self.m_glare = m_det
        self.det_box = np.array(det["box"], np.float32)  # tight box for this observed frame
        self.last_frame = frame_idx

        good = m_det >= TAU_HI
        self.good_streak = self.good_streak + 1 if good else 0
        if good:
            self.good_hits += 1
        if not self.is_confirmed and self.good_streak >= CONFIRM_HITS:
            self.is_confirmed = True

        if not self.is_confirmed:
            self.state = "TENTATIVE"
        elif good and (self.good_streak >= RELIABLE_DWELL or self.state == "RELIABLE"):
            self.state = "RELIABLE"
        else:
            self.state = "DEGRADED"
        self.mode = {"RELIABLE": "DET", "DEGRADED": "DEG"}.get(self.state, "TENTATIVE")

        if gap_before > REENTRY_GAP:
            # re-entry after a long gap: restart history so the velocity clamp
            # never sees a teleport step across the gap
            self.hist.clear()
            self.centers.clear()

        self.box = xyxy_to_norm_box(self.kf.get_state()[0], W, H)
        self.smooth = smooth_box(self.smooth, self.box)
        # the FUSED state goes to history with an honest m: the weighted Kalman
        # update already decided how much of the detection to trust
        self._push_hist(self.smooth, max(m_det, HIST_M_MIN))

    def update_missing(self, W, H, beacon_used=False):
        gap = int(self.kf.time_since_update)
        if gap == 1:
            if self.good_hits < FC_MIN_GOOD_HITS:
                # a young track has no trustworthy velocity: hold, don't coast
                self.kf.kf.x[4:7, 0] = 0.0
            else:
                # entering coast: glare-era updates can leave a bogus velocity.
                # The history median step is the trustworthy speed - clamp to it.
                v = self.kf.kf.x[4:6, 0]
                vmax = 2.0 * self.median_step_px(W, H)
                vn = float(np.hypot(v[0], v[1]))
                if vn > vmax > 0.0:
                    self.kf.kf.x[4:6, 0] = v * (vmax / vn)
        if gap >= FORECAST_START:
            # constant-velocity is not credible over long unobserved stretches
            self.kf.kf.x[4:6] *= VEL_DAMP
        self.box = xyxy_to_norm_box(self.kf.get_state()[0], W, H)
        if self.is_confirmed:
            self.state = "LOST"
            if beacon_used:
                self.mode = "BEACON"
            else:
                self.mode = "KALMAN"
        self.m_glare = float(GAMMA ** max(gap, 1))
        self.smooth = smooth_box(self.smooth, self.box)


class GlareTrack:
    def __init__(self):
        self.tracks = []
        self.frame_count = 0
        self._W = 1
        self._H = 1
        self._flare = 0.0
        self._flare_cool = 0
        KalmanBoxTracker.count = 0

    def _valid_detections(self, dets, W, H):
        """Two tiers: full-quality detections (associate + birth) and a
        BYTE-lite low-score tier that may only SUSTAIN a recently-seen track."""
        rows, metas, brows, bmetas = [], [], [], []
        for d in dets:
            if d["core_frac"] > ASSOC_CORE_MAX:
                continue  # near-pure light blob: not a measurement of a person
            if d["m_glare"] >= TAU_LO:
                xyxy = norm_box_to_xyxy(d["box"], W, H)
                score = max(TRACK_THRESH + 1e-3, min(1.0, d["conf"]))
                rows.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], score])
                metas.append(d)
            elif (d["m_glare"] >= BYTE_M_MIN
                  and d["core_frac"] <= BYTE_CORE_MAX):
                xyxy = norm_box_to_xyxy(d["box"], W, H)
                brows.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], max(0.05, d["conf"])])
                bmetas.append(d)
        det_arr = np.asarray(rows, np.float32) if rows else np.empty((0, 5), np.float32)
        byte_arr = np.asarray(brows, np.float32) if brows else np.empty((0, 5), np.float32)
        if len(det_arr) > 1:
            det_arr, metas = self._dedupe_dets(det_arr, metas)
        return det_arr, metas, byte_arr, bmetas

    def _dedupe_dets(self, det_arr, metas):
        order = np.argsort(-det_arr[:, 4])
        keep = []
        for i in order:
            b = det_arr[i, :4]
            duplicate = any(
                float(iou_batch(b[None, :], det_arr[j:j + 1, :4])[0, 0]) > 0.60
                for j in keep)
            if not duplicate:
                keep.append(int(i))
        keep = sorted(keep)
        return det_arr[keep], [metas[i] for i in keep]

    def _birth_allowed(self, det_row, meta):
        # Flare birth-guard (deployment): no new track while the frame is blooming
        # or in the brief refractory after, when the detector fires on the fading glow.
        if FLARE_BIRTH_GUARD > 0.0 and (self._flare > FLARE_BIRTH_GUARD
                                        or self._flare_cool > 0):
            return False
        # Fixed-camera scene veto: a birth whose box has ~no foreground residual sits
        # on a static structure / fixed light, not a worker who just entered -> reject
        # the high-conf detector hallucination at its source (default off).
        if SCENE_VETO and meta.get("bg_resid", float("inf")) < SCENE_VETO_THR:
            return False
        if meta["conf"] < BIRTH_CONF:
            return False
        if meta["m_glare"] < BIRTH_M_MIN:
            return False
        if meta["core_frac"] > BIRTH_CORE_MAX:
            return False  # mostly burnt core: a lamp, not a person
        bw, bh = float(meta["box"][2]), float(meta["box"][3])
        if bw < 0.006 or bh < 0.020:
            return False
        if bh / max(bw, 1e-6) < 0.80:
            return False
        dx1, dy1, dx2, dy2 = det_row[:4]
        dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dy2) * 0.5
        track_anchors = self.tracks if OVERLAP_BIRTH else []
        for t in track_anchors:
            refs = [t.kf.get_state()[0].astype(np.float32),
                    norm_box_to_xyxy(t.smooth, self._W, self._H)]
            for ref in refs:
                ref_iou = float(iou_batch(det_row[None, :4], ref[None, :])[0, 0])
                if ref_iou >= BIRTH_SUPPRESS_IOU:
                    return False
                rx1, ry1, rx2, ry2 = ref
                rcx, rcy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
                scale = max(dx2 - dx1, dy2 - dy1, rx2 - rx1, ry2 - ry1, 1.0)
                # Center-distance suppression kills duplicate births (a 2nd detector
                # box on ONE person: partial overlap, center-close). But applied
                # unconditionally it ALSO kills a DISTINCT person walking close with
                # ~zero overlap (the 211_88q failure: 2nd person blocked 364 frames).
                # So only center-suppress inside the duplicate OVERLAP BAND; a det
                # that is spatially separate (ref_iou < CENTER_MIN_IOU) is a different
                # object and is always allowed to birth. ref_iou==duplicate band keeps
                # the original anti-double-detection behaviour.
                if (ref_iou >= BIRTH_CENTER_MIN_IOU
                        and float(np.hypot(dcx - rcx, dcy - rcy)) < BIRTH_SUPPRESS_CENTER_FRAC * scale):
                    return False
        return True

    def _resolve_identity(self, matched, det_arr, trks, unmatched_trks):
        """Anti ID-steal/swap inside close groups - the price of no appearance.

        1. CONTESTED STEAL: a det matched to track ti while an overlapping
           hungry track tj fits nearly as well. Predictions drift but LAST
           OBSERVATIONS are anchored to real bodies, so the det is re-awarded
           to whichever track's last observation explains it best (tie -> the
           more recently seen track; an exiting track always loses to one that
           is still in the scene).
        2. PAIR SWAP: two matched dets whose assignment inverts the pair's
           stable historical order in BOTH x and foot-y get swapped back.
        """
        if len(matched) == 0 or len(self.tracks) < 2 or not len(det_arr):
            return matched, unmatched_trks
        pairs = [(int(d), int(t)) for d, t in matched]
        iou_dt = np.asarray(iou_batch(det_arr[:, :4], trks[:, :4]))
        iou_tt = np.asarray(iou_batch(trks[:, :4], trks[:, :4]))
        unmatched = set(int(t) for t in unmatched_trks)

        out = []
        for di, ti in pairs:
            winner = ti
            for tj in sorted(unmatched):
                if iou_tt[winner, tj] < AMBIG_IOU:
                    continue
                if self.tracks[tj].exiting and not self.tracks[winner].exiting:
                    continue  # a leaving track may never steal from one still present
                if iou_dt[di, tj] < iou_dt[di, winner] - AMBIG_MARGIN:
                    continue
                lo_w = self.tracks[winner].kf.last_observation[:4]
                lo_j = self.tracks[tj].kf.last_observation[:4]
                if lo_w[0] < 0 or lo_j[0] < 0:
                    continue
                iou_lo_w = float(iou_batch(det_arr[di:di + 1, :4],
                                           np.asarray(lo_w, np.float32)[None, :])[0, 0])
                iou_lo_j = float(iou_batch(det_arr[di:di + 1, :4],
                                           np.asarray(lo_j, np.float32)[None, :])[0, 0])
                w_exit = self.tracks[winner].exiting and not self.tracks[tj].exiting
                recency = (abs(iou_lo_j - iou_lo_w) <= 1e-6
                           and self.tracks[tj].kf.time_since_update
                           < self.tracks[winner].kf.time_since_update)
                steal = (iou_lo_j > iou_lo_w + 1e-6 or recency
                         or (w_exit and iou_lo_j > 0.1))
                if steal:
                    winner = tj
            if winner != ti:
                unmatched.discard(winner)
                unmatched.add(ti)
            out.append((di, winner))
        pairs = out

        def hist_med(t):
            c = np.asarray(list(self.tracks[t].centers)[-6:], np.float32)
            return float(np.median(c[:, 0])) * self._W, float(np.median(c[:, 1])) * self._H

        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                di, ti = pairs[a]
                dj, tj = pairs[b]
                if iou_tt[ti, tj] < AMBIG_IOU:
                    continue
                if len(self.tracks[ti].centers) < 3 or len(self.tracks[tj].centers) < 3:
                    continue
                hxi, hyi = hist_med(ti)
                hxj, hyj = hist_med(tj)
                dxi, dyi = (det_arr[di, 0] + det_arr[di, 2]) * 0.5, (det_arr[di, 1] + det_arr[di, 3]) * 0.5
                dxj, dyj = (det_arr[dj, 0] + det_arr[dj, 2]) * 0.5, (det_arr[dj, 1] + det_arr[dj, 3]) * 0.5
                w_ref = max(det_arr[di, 2] - det_arr[di, 0], det_arr[dj, 2] - det_arr[dj, 0], 1.0)
                h_ref = max(det_arr[di, 3] - det_arr[di, 1], det_arr[dj, 3] - det_arr[dj, 1], 1.0)
                flip_x = (abs(hxi - hxj) > SWAP_SEP_FRAC * w_ref
                          and abs(dxi - dxj) > SWAP_SEP_FRAC * w_ref
                          and np.sign(dxi - dxj) == -np.sign(hxi - hxj))
                flip_y = (abs(hyi - hyj) > SWAP_SEP_FRAC * h_ref
                          and abs(dyi - dyj) > SWAP_SEP_FRAC * h_ref
                          and np.sign(dyi - dyj) == -np.sign(hyi - hyj))
                if flip_x and flip_y:
                    pairs[a] = (di, tj)
                    pairs[b] = (dj, ti)
        return pairs, np.asarray(sorted(unmatched), dtype=int)

    def _match_ok(self, di, ti, det_arr, det_meta):
        """Re-entry gate: a long-lost track must not be revived by a ghost."""
        mt = self.tracks[ti]
        gap = int(mt.kf.time_since_update)  # after predict: >= 1
        if mt.exiting and gap > EXIT_REASSOC_GAP:
            # the person walked out: whatever is detected near the doorway now
            # is somebody else
            return False
        if gap <= REENTRY_GAP:
            return True
        meta = det_meta[di]
        if meta["m_glare"] < REENTRY_M_MIN:
            return False
        if meta["core_frac"] > BIRTH_CORE_MAX:
            return False
        px1, py1, px2, py2 = [float(v) for v in mt.kf.get_state()[0][:4]]
        diag = float(np.hypot(px2 - px1, py2 - py1))
        dx1, dy1, dx2, dy2 = det_arr[di][:4]
        dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dy2) * 0.5
        pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        limit = max(diag, 1.5 * mt.median_step_px(self._W, self._H) * gap + 0.5 * diag)
        return float(np.hypot(dcx - pcx, dcy - pcy)) <= limit

    def _apply_det(self, mt, det_row, meta, frame_idx):
        """Confidence-weighted Kalman update: m scales the measurement noise."""
        gap_before = int(mt.kf.time_since_update)
        scale = 1.0 / max(float(meta["m_glare"]), R_M_FLOOR)
        mt.kf.kf.R = mt.R0 * scale
        mt.kf.update(det_row)
        mt.kf.kf.R = mt.R0
        mt.update_detection(meta, self._W, self._H, frame_idx, gap_before)

    @staticmethod
    def _weak_center_update(kfin, zx, zy, r_x, r_y):
        """Manual center-only Kalman update on (x, P). Deliberately bypasses
        KalmanFilterNew.update so the beacon never touches history_obs /
        freeze-unfreeze (it is a weak cue, not an observation)."""
        Hm = kfin.H[:2, :]
        R = np.diag([r_x, r_y])
        z = np.array([[zx], [zy]], np.float64)
        y = z - Hm @ kfin.x
        PHT = kfin.P @ Hm.T
        S = Hm @ PHT + R
        K = PHT @ np.linalg.inv(S)
        kfin.x = kfin.x + K @ y
        I_KH = np.eye(kfin.dim_x) - K @ Hm
        kfin.P = I_KH @ kfin.P @ I_KH.T + K @ R @ K.T

    def _try_beacon(self, mt, core_fn, W, H):
        """If a compact lamp core sits in the predicted HEAD BAND of an
        unobserved confirmed track, use it as a weak center pseudo-measurement.
        The lamp serves an existing person; it can never create one.

        Note on the beacon point: we use the ANALYTIC core centroid
        (``core_centroid_in_box``) with a fixed isotropic noise. The analytic core
        is the same photometric signature on real and synthetic frames
        (domain-invariant), which keeps the beacon transfer-safe. The controller
        consumes solely ``m_glare`` from ``h_psi``."""
        core, S = core_fn()
        if core is None:
            return False
        x1, y1, x2, y2 = [float(v) for v in mt.kf.get_state()[0][:4]]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            return False
        sx, sy = S / float(W), S / float(H)
        roi = ((x1 - 0.10 * bw) * sx, (y1 - BEACON_EXPAND_UP * bh) * sy,
               (x2 + 0.10 * bw) * sx, (y1 + BEACON_HEAD_BAND * bh) * sy)
        if mask_area_inside(core, roi) < BEACON_MIN_PX:
            return False
        mu = core_centroid_in_box(core, roi)
        if mu is None:
            return False
        u, v = float(mu[0]) / sx, float(mu[1]) / sy
        # lamp is on the head: body center is roughly 0.35*h below it
        zx, zy = u, v + 0.35 * bh
        pcx, pcy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if float(np.hypot(zx - pcx, zy - pcy)) > BEACON_GATE_FRAC * float(np.hypot(bw, bh)):
            return False
        r_x = max(25.0, (0.08 * bh) ** 2)
        self._weak_center_update(mt.kf.kf, zx, zy, r_x, 4.0 * r_x)
        mt.beacon = (u, v)
        return True

    def step(self, dets, W, H, frame_idx, core_fn=None, flare=0.0):
        self.frame_count += 1
        self._W = W
        self._H = H
        self._flare = float(flare)
        if FLARE_BIRTH_GUARD > 0.0:
            if self._flare > FLARE_BIRTH_GUARD:
                self._flare_cool = FLARE_BIRTH_COOL   # arm the refractory
            elif self._flare_cool > 0:
                self._flare_cool -= 1
        det_arr, det_meta, byte_arr, byte_meta = self._valid_detections(dets, W, H)
        det_arr = det_arr[det_arr[:, 4] > TRACK_THRESH] if len(det_arr) else det_arr

        trks = np.zeros((len(self.tracks), 5), np.float32)
        to_del = []
        for i, mt in enumerate(self.tracks):
            mt.age_frames += 1
            pos = mt.kf.predict()[0]
            trks[i] = [pos[0], pos[1], pos[2], pos[3], 0.0]
            if np.any(np.isnan(pos)):
                to_del.append(i)
        for i in reversed(to_del):
            self.tracks.pop(i)
        if to_del:
            trks = np.delete(trks, to_del, axis=0)

        velocities = np.array(
            [t.kf.velocity if t.kf.velocity is not None else np.array((0.0, 0.0))
             for t in self.tracks])
        last_boxes = np.array([t.kf.last_observation for t in self.tracks])
        k_obs = np.array([k_previous_obs(t.kf.observations, t.kf.age, DELTA_T)
                          for t in self.tracks])

        matched, unmatched_dets, unmatched_trks = oc_associate(
            det_arr, trks, IOU_THRESH, velocities, k_obs, INERTIA)
        matched, unmatched_trks = self._resolve_identity(matched, det_arr, trks, unmatched_trks)
        rej_d, rej_t = [], []
        for di, ti in matched:
            di, ti = int(di), int(ti)
            if self._match_ok(di, ti, det_arr, det_meta):
                self._apply_det(self.tracks[ti], det_arr[di], det_meta[di], frame_idx)
            else:
                rej_d.append(di)
                rej_t.append(ti)
        if rej_d:
            unmatched_dets = np.union1d(unmatched_dets, np.asarray(rej_d, dtype=int)).astype(int)
            unmatched_trks = np.union1d(unmatched_trks, np.asarray(rej_t, dtype=int)).astype(int)

        # OCR second round on last observations
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = det_arr[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = np.asarray(iou_batch(left_dets, left_trks))
            if iou_left.size and iou_left.max() > IOU_THRESH:
                rematched = linear_assignment(-iou_left)
                rm_d, rm_t = [], []
                for li, lt in rematched:
                    if iou_left[li, lt] < IOU_THRESH:
                        continue
                    di, ti = int(unmatched_dets[li]), int(unmatched_trks[lt])
                    if not self._match_ok(di, ti, det_arr, det_meta):
                        continue
                    self._apply_det(self.tracks[ti], det_arr[di], det_meta[di], frame_idx)
                    rm_d.append(di)
                    rm_t.append(ti)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.asarray(rm_d, dtype=int))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.asarray(rm_t, dtype=int))

        # BYTE-lite third round: low-score dets keep RECENT tracks alive (no birth)
        if len(byte_arr) and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_b = np.asarray(iou_batch(byte_arr, u_trks))
            if iou_b.size and iou_b.max() > BYTE_IOU:
                bm = linear_assignment(-iou_b)
                used_t = []
                for bi, lt in bm:
                    if iou_b[bi, lt] < BYTE_IOU:
                        continue
                    ti = int(unmatched_trks[lt])
                    mt = self.tracks[ti]
                    if int(mt.kf.time_since_update) > REENTRY_GAP:
                        continue  # long-lost tracks need the strict re-entry gate
                    if mt.exiting:
                        continue  # a weak det at the border is not the person who left
                    self._apply_det(mt, byte_arr[int(bi)], byte_meta[int(bi)], frame_idx)
                    used_t.append(ti)
                if used_t:
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.asarray(used_t, dtype=int))

        for ti in unmatched_trks:
            mt = self.tracks[int(ti)]
            mt.kf.update(None)
            mt.beacon = None
            beacon_used = False
            exiting = mt.update_exit_state(W, H) if mt.is_confirmed else False
            if core_fn is not None and mt.is_confirmed and not exiting:
                # an exiting track must not be kept alive by someone else's lamp
                beacon_used = self._try_beacon(mt, core_fn, W, H)
            mt.update_missing(W, H, beacon_used=beacon_used)

        for di in unmatched_dets:
            d = det_meta[int(di)]
            if not self._birth_allowed(det_arr[int(di)], d):
                continue
            kf = KalmanBoxTracker(det_arr[int(di)], delta_t=DELTA_T)
            # OC-SORT's constructor seeds the state but does not count the
            # initial detection as a hit; mark it once without a double update.
            kf.last_observation = det_arr[int(di)].copy()
            kf.observations[kf.age] = det_arr[int(di)].copy()
            kf.history_observations.append(det_arr[int(di)].copy())
            kf.hits = 1
            kf.hit_streak = 1
            self.tracks.append(MotionTrack(kf, d, W, H, frame_idx))

        alive = []
        for t in self.tracks:
            gap = int(t.kf.time_since_update)
            if not t.is_confirmed and t.age_frames > TENTATIVE_MAX_AGE:
                continue  # never proved to be a person
            if gap > MAX_AGE:
                continue  # hard cap (covers the longest observed glare gap)
            keep = gap <= t.allowed_gap()
            if not keep:
                continue  # fill budget exhausted / exit gate fired
            alive.append(t)
        self.tracks = alive

        # Emit a confirmed track while it has been observed within OUTPUT_MAX_GAP frames.
        # A box held on pure prediction far past the last observation is the dominant
        # false-positive source on a strong detector (the person has usually just left or
        # been briefly missed, not burnt), so the short output coast is the precision lever.
        def _emit_ok(t):
            if not t.is_confirmed:
                return False
            gap = int(t.kf.time_since_update)
            if gap <= OUTPUT_MAX_GAP:
                return True
            # beacon-anchored holds are backed by a live lamp measurement, so they
            # may coast further (bridges glare flares); pure Kalman drift may not.
            return t.mode == "BEACON" and gap <= OUTPUT_MAX_GAP_BEACON

        outputs = [t for t in self.tracks if _emit_ok(t)]
        self._update_vis(outputs, W, H)
        return outputs

    def _update_vis(self, tracks, W, H):
        if not tracks:
            return
        boxes, foots = [], []
        for t in tracks:
            x1, y1, x2, y2 = norm_box_to_xyxy(t.smooth if t.mode != "DET" else t.box, W, H)
            boxes.append((x1, y1, x2, y2))
            foots.append(y2)
        vis = compute_vis_geom(boxes, foots, tau_ov=0.2, dmin=8.0)
        for t, v in zip(tracks, vis):
            t.vis_geom = float(v)
            if t.state in {"RELIABLE", "DEGRADED"}:
                raw = t.m_glare if t.m_glare >= TAU_HI else (t.m_glare * 0.5)
                t.m = float(np.clip(raw * t.vis_geom, 0.0, 1.0))
            else:
                t.m = float(np.clip((GAMMA ** max(t.kf.time_since_update, 1)) * t.vis_geom,
                                    0.0, 1.0))


# Distinct per-identity colors (BGR). GMOT_DRAW_BY_ID=1 (default) colors each box
# by its track id like a standard MOT viewer; set to 0 to recover the legacy
# reliability colouring (green=observed, red=low-m).
DRAW_BY_ID = os.environ.get("GMOT_DRAW_BY_ID", "1") not in ("0", "off", "false", "")
_ID_PALETTE = [
    (80, 220, 80), (60, 200, 255), (255, 120, 60), (200, 80, 255), (60, 255, 255),
    (255, 200, 40), (40, 120, 255), (180, 255, 120), (255, 80, 200), (120, 220, 255),
    (90, 160, 255), (210, 160, 60), (160, 90, 255), (60, 230, 160), (230, 230, 90),
]


def id_color(tid):
    return _ID_PALETTE[int(tid) % len(_ID_PALETTE)]


def draw_track(frame, t, W, H):
    x1, y1, x2, y2 = [int(round(v)) for v in norm_box_to_xyxy(t.smooth, W, H)]
    col = id_color(t.id) if DRAW_BY_ID else reliability_color(t.m)
    thick = 2 if t.mode in {"KALMAN", "BEACON", "FORECAST"} else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
    if t.beacon is not None:
        cv2.circle(frame, (int(round(t.beacon[0])), int(round(t.beacon[1]))), 4, (0, 0, 255), 1,
                   cv2.LINE_AA)  # RED lamp-core beacon (matches teaser Fig. 1)
    label = f"ID {t.id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), col, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def write_mot_row(rows, frame_idx, t, W, H):
    x1, y1, x2, y2 = norm_box_to_xyxy(t.smooth, W, H)
    rows.append([frame_idx, t.id, max(0.0, x1), max(0.0, y1),
                 max(1.0, x2 - x1), max(1.0, y2 - y1), max(0.001, t.m), 1, 1])


def detector_debug(seq_dir, percep, out_csv, max_frames=0, frame_stride=1):
    """Dump per-detection perception stats (used by hard-negative mining)."""
    img_paths = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    if max_frames:
        img_paths = img_paths[:max_frames]
    indexed = list(enumerate(img_paths, start=1))
    if frame_stride > 1:
        indexed = indexed[::frame_stride]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "det", "conf", "m_glare", "core_frac",
                    "cx", "cy", "bw", "bh"])
        for fi, ip in indexed:
            frame = cv2.imread(ip)
            for di, d in enumerate(percep(frame)):
                b = d["box"]
                w.writerow([fi, di, d["conf"], d["m_glare"],
                            d["core_frac"], b[0], b[1], b[2], b[3]])
    print(f"detector-debug saved -> {out_csv}")


def run_sequence(seq_dir, percep, out_path, max_frames=0, mot_out=None):
    """Track a frame-folder sequence; writes an annotated mp4 (+ optional MOT txt)."""
    img_paths = sorted(glob.glob(os.path.join(seq_dir, "images", "*.jpg")))
    if max_frames:
        img_paths = img_paths[:max_frames]
    if not img_paths:
        raise FileNotFoundError(f"no frames in {seq_dir}")
    H, W = cv2.imread(img_paths[0]).shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 22.0, (W, H))
    tracker = GlareTrack()
    if hasattr(percep, "reset_scene"):
        percep.reset_scene()
    mot_rows = []
    stats = {}
    n, t_proc = 0, 0.0

    # prefetch thread: decode the next frames while the GPU works on this one
    rq: "queue.Queue" = queue.Queue(maxsize=8)

    def _reader():
        for p in img_paths:
            rq.put(cv2.imread(p))
        rq.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    frame_idx = 0
    while True:
        frame = rq.get()
        if frame is None:
            break
        frame_idx += 1
        t0 = time.time()
        dets = percep(frame)
        tracks = tracker.step(dets, W, H, frame_idx, core_fn=percep.core_mask,
                              flare=getattr(percep, "flare_frac", 0.0))
        for t in tracks:
            stats[t.mode] = stats.get(t.mode, 0) + 1
            draw_track(frame, t, W, H)
            if mot_out is not None:
                write_mot_row(mot_rows, frame_idx, t, W, H)
        t_proc += time.time() - t0
        n += 1
        writer.write(frame)
        if n % 100 == 0:
            print(f"  frame {n}: {n / max(t_proc, 1e-6):.1f} FPS avg | tracks {len(tracks)}",
                  flush=True)
    writer.release()

    if mot_out is not None:
        os.makedirs(os.path.dirname(mot_out), exist_ok=True)
        with open(mot_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for r in mot_rows:
                w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}",
                            f"{r[5]:.2f}", f"{r[6]:.4f}", r[7], r[8]])

    avg = n / max(t_proc, 1e-6)
    print(f"\nDONE {n} frames | avg {avg:.1f} FPS "
          f"{'REALTIME OK' if avg >= 22 else '(below 22 fps source)'}")
    print("frame-modes: " + " ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"saved -> {out_path}")


def run_video(video_path, percep, out_path, max_frames=0, mot_out=None,
              debug_csv=None):
    """Track a raw .mp4; decode and encode run on their own threads so the GPU
    loop never waits on I/O."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 22.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError(f"bad video dimensions: {video_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tracker = GlareTrack()
    if hasattr(percep, "reset_scene"):
        percep.reset_scene()
    mot_rows, dbg_rows = [], []
    stats = {}
    n, t_proc = 0, 0.0

    rq: "queue.Queue" = queue.Queue(maxsize=8)
    wq: "queue.Queue" = queue.Queue(maxsize=8)

    def _reader():
        while True:
            ok_, fr = cap.read()
            if not ok_:
                rq.put(None)
                return
            rq.put(fr)

    def _writer():
        while True:
            fr = wq.get()
            if fr is None:
                return
            writer.write(fr)

    t_read = threading.Thread(target=_reader, daemon=True)
    t_write = threading.Thread(target=_writer, daemon=True)
    t_read.start()
    t_write.start()

    while True:
        frame = rq.get()
        if frame is None:
            break
        frame_idx = n + 1
        t0 = time.time()
        dets = percep(frame)
        tracks = tracker.step(dets, W, H, frame_idx, core_fn=percep.core_mask,
                              flare=getattr(percep, "flare_frac", 0.0))
        for t in tracks:
            stats[t.mode] = stats.get(t.mode, 0) + 1
            draw_track(frame, t, W, H)
            if mot_out is not None:
                write_mot_row(mot_rows, frame_idx, t, W, H)
            if debug_csv is not None:
                vx, vy = float(t.kf.kf.x[4, 0]), float(t.kf.kf.x[5, 0])
                dbg_rows.append(
                    f"{frame_idx},{t.id},{t.mode},{t.state},{t.smooth[0]:.4f},"
                    f"{t.smooth[1]:.4f},{t.smooth[2]:.4f},{t.smooth[3]:.4f},"
                    f"{t.m:.3f},{int(t.kf.time_since_update)},{vx:.2f},{vy:.2f}")
        t_proc += time.time() - t0
        n += 1
        wq.put(frame)
        if n % 100 == 0:
            print(f"  frame {n}: {n / max(t_proc, 1e-6):.1f} FPS avg | tracks {len(tracks)}",
                  flush=True)
        if max_frames and n >= max_frames:
            break
    wq.put(None)
    t_write.join(timeout=30)
    cap.release()
    writer.release()

    if mot_out is not None:
        os.makedirs(os.path.dirname(mot_out), exist_ok=True)
        with open(mot_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for r in mot_rows:
                w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}",
                            f"{r[5]:.2f}", f"{r[6]:.4f}", r[7], r[8]])
    if debug_csv is not None:
        os.makedirs(os.path.dirname(debug_csv), exist_ok=True)
        with open(debug_csv, "w", encoding="utf-8") as f:
            f.write("frame,id,mode,state,cx,cy,w,h,m,gap,vx,vy\n")
            f.write("\n".join(dbg_rows) + "\n")

    avg = n / max(t_proc, 1e-6)
    print(f"\nDONE {n} frames | avg {avg:.1f} FPS (source {fps:.0f})")
    print("frame-modes: " + " ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--seq", default=None,
                    help="sequence folder name in the tracking dataset; "
                         "default = a random test sequence")
    ap.add_argument("--trk_root", default=None,
                    help="override root holding <seq>/images frame folders")
    ap.add_argument("--video", default=None, help="raw video path (no GT needed)")
    ap.add_argument("--detector", default=None, help="override detector weights")
    ap.add_argument("--hpsi", default=None, help="override h_psi head weights")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mot_out", default=None)
    ap.add_argument("--debug_csv", default=None)
    ap.add_argument("--detector_debug", action="store_true",
                    help="dump per-detection stats CSV instead of tracking")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="detector_debug only: process every k-th frame")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if (args.device == "cuda"
                        or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"device: {device} | motion: Kalman constant-velocity (no forecaster)")

    best = args.detector or weight_path(cfg, "detector")
    head = args.hpsi or weight_path(cfg, "hpsi")
    percep = Perception(best, head, device, args.conf)
    out_dir = results_dir(cfg, "tracking")

    if args.video is not None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = args.out or os.path.join(out_dir, f"{base}_track.mp4")
        print(f"tracking raw video {args.video} -> {out_path}")
        run_video(args.video, percep, out_path, max_frames=args.max_frames,
                  mot_out=args.mot_out, debug_csv=args.debug_csv)
        return

    trk_root = args.trk_root or data_path(cfg, "tracking")
    if args.seq is None:
        args.seq = random.Random(args.seed).choice(make_splits()["test"])
        print(f"random test sequence -> {args.seq}")
    seq_dir = os.path.join(trk_root, args.seq)

    if args.detector_debug:
        out_csv = args.out or os.path.join(out_dir, f"{args.seq}_detector_debug.csv")
        detector_debug(seq_dir, percep, out_csv, max_frames=args.max_frames,
                       frame_stride=args.frame_stride)
        return

    out_path = args.out or os.path.join(out_dir, f"{args.seq}_track.mp4")
    print(f"tracking {args.seq} -> {out_path}")
    run_sequence(seq_dir, percep, out_path, max_frames=args.max_frames,
                 mot_out=args.mot_out)


if __name__ == "__main__":
    main()
