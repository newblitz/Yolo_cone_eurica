import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from distance import CameraModel, DistanceEstimator


def synthetic_box(camera, distance_m, height_m=0.70):
    h_px = camera.fy * height_m / distance_m
    y_bottom = camera.cy + camera.fy * camera.cam_height_m / distance_m
    return (600.0, y_bottom - h_px, 700.0, y_bottom)


def test_synthetic_cone_recovers_distance():
    camera = CameraModel.from_hfov(1280, 720, 65.0, cam_height_m=1.4)
    estimate = DistanceEstimator(camera).estimate("cone", synthetic_box(camera, 10.0))
    assert abs(estimate.distance_m - 10.0) < 0.1


def test_bottom_edge_disables_ground_method():
    camera = CameraModel.from_hfov(1280, 720, 65.0)
    box = (600.0, 600.0, 700.0, 720.0)
    estimate = DistanceEstimator(camera).estimate("cone", box)
    assert estimate.z_ground is None


def test_box_above_horizon_has_no_ground_estimate():
    camera = CameraModel.from_hfov(1280, 720, 65.0)
    estimate = DistanceEstimator(camera).estimate("cone", (600.0, 100.0, 700.0, 150.0))
    assert estimate.z_ground is None


def test_disagreement_is_low_confidence():
    camera = CameraModel.from_hfov(1280, 720, 65.0)
    # A very low box makes the ground estimate much shorter than the size estimate.
    estimate = DistanceEstimator(camera).estimate("cone", (600.0, 400.0, 700.0, 700.0))
    assert "disagree" in estimate.method
    assert estimate.confidence == "low"
