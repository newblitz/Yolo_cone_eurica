"""Monocular distance estimation from a bounding box, purely geometric.

Two estimators are fused:

  M1  known-size pinhole       Z = fy * H_real / h_px
        needs: focal length (px) and the real height of the object class.
        works for: every class, but breaks when the real size varies or the box is truncated.

  M2  ground-plane geometry    D = h_cam / tan(alpha),   alpha = atan((y_bottom - cy) / fy) - pitch
        needs: camera height above ground and (optionally) camera pitch.
        works for: objects whose box bottom touches the ground (cones, barriers).
        NOT used for stop signs: the bottom of a sign box is on a pole, not on the road.

Fusion: each method gets a relative uncertainty (sigma) propagated from pixel noise, prior-size
error, pitch error and camera-height error. They are combined by inverse-variance weighting in
log-space. If the two disagree by more than `disagree_ratio`, the more certain one is used and
the result is flagged low-confidence.

Conventions: pitch_deg > 0 means the camera is tilted UP. Distances are forward distance in metres.
All parameters below are ASSUMPTIONS unless you calibrate them - document them in the README.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- priors
@dataclass(frozen=True)
class ObjectPrior:
    height_m: float        # real-world height of the detected BOX (sign face, not the whole pole)
    height_rel_sigma: float  # relative uncertainty of that height
    grounded: bool         # True if the bottom edge of the box touches the road


# Edit these to match your region. Sources of variation: cones come in 0.45 / 0.7 / 0.9 m,
# stop-sign faces in 0.6 / 0.75 / 0.9 m, barriers in 0.8-1.2 m.
DEFAULT_PRIORS: dict[str, ObjectPrior] = {
    "cone": ObjectPrior(0.70, 0.15, True),
    "barrier": ObjectPrior(0.90, 0.20, True),
    "stop_sign": ObjectPrior(0.75, 0.10, False),
}


# --------------------------------------------------------------------------- camera
@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    cam_height_m: float = 1.4          # lens height above the road
    cam_height_sigma_m: float = 0.10
    pitch_deg: float = 0.0             # > 0: tilted up
    pitch_sigma_deg: float = 1.0

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float = 65.0, **kw) -> "CameraModel":
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, f, f, width / 2, height / 2, **kw)

    @classmethod
    def from_focal(cls, width: int, height: int, focal_px: float, ref_width: int = 1280, **kw) -> "CameraModel":
        """focal_px was measured at an image width of ref_width; rescale to the actual width."""
        f = focal_px * width / ref_width
        return cls(width, height, f, f, width / 2, height / 2, **kw)

    @property
    def horizon_y(self) -> float:
        return self.cy + self.fy * math.tan(math.radians(self.pitch_deg))


def calibrate_focal_px(box_height_px: float, real_height_m: float, true_distance_m: float) -> float:
    """One tape-measured photo of an object of known height gives you the focal length in px."""
    return true_distance_m * box_height_px / real_height_m


# --------------------------------------------------------------------------- result
@dataclass
class DistanceEstimate:
    distance_m: Optional[float]
    z_height: Optional[float] = None    # M1
    z_ground: Optional[float] = None    # M2
    rel_sigma: Optional[float] = None
    method: str = "none"                # "fused" | "height" | "ground" | "height(disagree)" | "ground(disagree)"
    confidence: str = "none"            # "high" | "medium" | "low" | "none"
    lateral_m: Optional[float] = None   # + = right of the camera axis
    truncated: bool = False


# --------------------------------------------------------------------------- estimator
class DistanceEstimator:
    def __init__(self, camera: CameraModel, priors: Optional[dict] = None,
                 disagree_ratio: float = 1.4, border_px: float = 2.0):
        self.cam = camera
        self.priors = priors or DEFAULT_PRIORS
        self.disagree_ratio = disagree_ratio
        self.border_px = border_px

    # ---- M1
    def _height_method(self, prior: ObjectPrior, h_px: float, trunc_top: bool, trunc_bottom: bool):
        z = self.cam.fy * prior.height_m / h_px
        sig_px = 2.0 + 0.03 * h_px                     # edge jitter of a detector box
        rel = math.hypot(prior.height_rel_sigma, sig_px / h_px)
        if trunc_top or trunc_bottom:                  # box height under-estimates the object
            rel *= 3.0
        return z, rel

    # ---- M2
    def _ground_method(self, y_bottom: float, box_h_px: float):
        cam = self.cam
        alpha = math.atan2(y_bottom - cam.cy, cam.fy) - math.radians(cam.pitch_deg)
        if alpha < math.radians(0.3):                  # at/above the horizon -> ill-posed
            return None
        d = cam.cam_height_m / math.tan(alpha)
        sig_pix = 2.0 + 0.02 * box_h_px
        sig_alpha = math.hypot(sig_pix / cam.fy, math.radians(cam.pitch_sigma_deg))
        rel = math.hypot(2 * sig_alpha / math.sin(2 * alpha), cam.cam_height_sigma_m / cam.cam_height_m)
        return d, rel

    def estimate(self, class_name: str, box_xyxy) -> DistanceEstimate:
        prior = self.priors.get(class_name)
        if prior is None:
            return DistanceEstimate(None)
        x1, y1, x2, y2 = box_xyxy
        h_px = y2 - y1
        if h_px < 2:
            return DistanceEstimate(None)
        cam, b = self.cam, self.border_px
        trunc_top, trunc_bottom = y1 <= b, y2 >= cam.height - b

        z1, s1 = self._height_method(prior, h_px, trunc_top, trunc_bottom)
        m2 = None
        if prior.grounded and not trunc_bottom:
            m2 = self._ground_method(y2, h_px)
        z2, s2 = m2 if m2 else (None, None)

        if z2 is None:
            z, s, method = z1, s1, "height"
        else:
            if max(z1, z2) / min(z1, z2) > self.disagree_ratio:
                (z, s) = (z1, s1) if s1 <= s2 else (z2, s2)
                method = "height(disagree)" if s1 <= s2 else "ground(disagree)"
            else:
                w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
                z = math.exp((w1 * math.log(z1) + w2 * math.log(z2)) / (w1 + w2))
                s, method = math.sqrt(1 / (w1 + w2)), "fused"

        if "disagree" in method or s >= 0.30:
            conf = "low"
        elif s >= 0.15:
            conf = "medium"
        else:
            conf = "high"
        xc = (x1 + x2) / 2
        return DistanceEstimate(
            distance_m=z, z_height=z1, z_ground=z2, rel_sigma=s, method=method, confidence=conf,
            lateral_m=(xc - cam.cx) * z / cam.fx, truncated=trunc_top or trunc_bottom,
        )


def format_label(display_name: str, est: DistanceEstimate) -> str:
    """'Cone, 1.5m' (the format required by the assignment)."""
    return display_name if est.distance_m is None else f"{display_name}, {est.distance_m:.1f}m"


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    cam = CameraModel.from_hfov(1280, 720, hfov_deg=65, cam_height_m=1.4)
    est = DistanceEstimator(cam)
    print(f"focal = {cam.fx:.0f}px, horizon row = {cam.horizon_y:.0f}")
    # A 0.7 m cone, 10 m away: h_px = f*H/Z ; bottom row = cy + f*h_cam/Z
    z_true = 10.0
    h_px = cam.fy * 0.70 / z_true
    y_bot = cam.cy + cam.fy * cam.cam_height_m / z_true
    box = (600, y_bot - h_px, 600 + 0.5 * h_px, y_bot)
    r = est.estimate("cone", box)
    print(f"cone  true={z_true}m -> {r}")
    print(format_label("Cone", r))
    sign_h = cam.fy * 0.75 / 12.0
    r = est.estimate("stop_sign", (700, 200, 700 + sign_h, 200 + sign_h))
    print(f"sign  true=12m  -> {format_label('Stop Sign', r)}  ({r.method}, {r.confidence})")
