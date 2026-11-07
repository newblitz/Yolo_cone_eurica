"""Distance-estimate sensitivity to camera and object-size assumptions."""
from __future__ import annotations

from distance import CameraModel, DistanceEstimator


DISTANCES = (3.0, 6.0, 10.0, 15.0)


def estimate(distance_m: float, focal_scale: float = 1.0, camera_height_m: float = 1.4,
             pitch_deg: float = 0.0, object_height_scale: float = 1.0) -> float:
    baseline_camera = CameraModel.from_hfov(1280, 720, 65.0, cam_height_m=1.4, pitch_deg=0.0)
    baseline_prior = DistanceEstimator(baseline_camera).priors["cone"]
    h_px = baseline_camera.fy * baseline_prior.height_m / distance_m
    y_bottom = baseline_camera.cy + baseline_camera.fy * baseline_camera.cam_height_m / distance_m
    box = (600.0, y_bottom - h_px, 700.0, y_bottom)
    cam = CameraModel.from_hfov(
        1280, 720, 65.0 * 1.0 / focal_scale,
        cam_height_m=camera_height_m, pitch_deg=pitch_deg,
    )
    priors = dict(DistanceEstimator(cam).priors)
    prior = priors["cone"]
    priors["cone"] = type(prior)(prior.height_m * object_height_scale, prior.height_rel_sigma, prior.grounded)
    est = DistanceEstimator(cam, priors=priors)
    return est.estimate("cone", box).distance_m


def main() -> None:
    scenarios = {
        "focal length -10%": dict(focal_scale=0.9),
        "focal length +10%": dict(focal_scale=1.1),
        "camera height -0.1m": dict(camera_height_m=1.3),
        "camera height +0.1m": dict(camera_height_m=1.5),
        "pitch -1deg": dict(pitch_deg=-1.0),
        "pitch +1deg": dict(pitch_deg=1.0),
        "object height -15%": dict(object_height_scale=0.85),
        "object height +15%": dict(object_height_scale=1.15),
    }
    print("| Assumption change | 3m | 6m | 10m | 15m |")
    print("|---|---:|---:|---:|---:|")
    for label, kwargs in scenarios.items():
        changes = []
        for distance in DISTANCES:
            baseline = estimate(distance)
            changed = estimate(distance, **kwargs)
            changes.append(f"{(changed / baseline - 1) * 100:+.1f}%")
        print(f"| {label} | " + " | ".join(changes) + " |")


if __name__ == "__main__":
    main()
