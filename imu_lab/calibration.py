"""Calibration methods adapted from the user's FT232H/MPU-6050 notebook.

Raw accelerometer values are in g; raw gyro values in degree/second.
We retain the original order: accel ellipsoid -> six-pose body alignment,
gyro stationary offset -> gyro axes transform. Output axes MUST agree.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Calibration:
    accel_bias_g: np.ndarray
    accel_matrix: np.ndarray
    accel_body_rotation: np.ndarray
    gyro_bias_dps: np.ndarray
    gyro_matrix: np.ndarray

    @classmethod
    def identity(cls) -> "Calibration":
        return cls(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3), np.eye(3))

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf8"))
        return cls(
            np.asarray(data["accel_bias_g"], dtype=float),
            np.asarray(data["accel_matrix"], dtype=float),
            np.asarray(data["accel_body_rotation"], dtype=float),
            np.asarray(data["gyro_bias_dps"], dtype=float),
            np.asarray(data["gyro_matrix"], dtype=float),
        ).validate()

    def validate(self) -> "Calibration":
        for name in ("accel_bias_g", "gyro_bias_dps"):
            v = getattr(self, name)
            if v.shape != (3,) or not np.isfinite(v).all():
                raise ValueError(f"{name} must contain three finite numbers")
        for name in ("accel_matrix", "accel_body_rotation", "gyro_matrix"):
            a = getattr(self, name)
            if a.shape != (3, 3) or not np.isfinite(a).all() or abs(np.linalg.det(a)) < 1e-8:
                raise ValueError(f"{name} must be a finite, nonsingular 3x3 matrix")
        if (np.linalg.norm(self.accel_body_rotation.T @ self.accel_body_rotation - np.eye(3)) > 1e-2
                or np.linalg.det(self.accel_body_rotation) <= 0):
            raise ValueError("accel_body_rotation must be a proper 3D rotation")
        return self

    def to_json(self, path: str | Path) -> None:
        self.validate()
        data = {name: getattr(self, name).tolist() for name in (
            "accel_bias_g", "accel_matrix", "accel_body_rotation", "gyro_bias_dps", "gyro_matrix")}
        Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        """Convert RAW g and dps to CALIBRATED g and dps in one body frame.

        The input dictionary is not mutated. This should run exactly once:
        the preintegration code must NOT subtract these biases again.
        """
        a = self.accel_body_rotation @ (self.accel_matrix @ (
            np.asarray(record["a"], dtype=float) - self.accel_bias_g))
        w = self.gyro_matrix @ (np.asarray(record["w"], dtype=float) - self.gyro_bias_dps)
        return {**record, "a": a.tolist(), "w": w.tolist()}


def calibration_callback(raw_data: dict, calibration: Calibration) -> dict:
    return calibration.apply(raw_data)


def fit_accelerometer_ellipsoid(acc_g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit ||L(a - b)|| = 1g to MANY stationary orientations.

    This is the same triangular-L model as the source calibration notebook.
    Motion-contaminated samples break the static-gravity assumption.
    """
    A = np.asarray(acc_g, float)
    if A.ndim != 2 or A.shape[1] != 3 or len(A) < 30:
        raise ValueError("Provide at least 30 3-axis static samples across many orientations")
    if np.linalg.matrix_rank(A - A.mean(axis=0)) < 3:
        raise ValueError("Insufficient orientation diversity for a 3D ellipsoid fit")
    def unpack(params):
        bias = params[:3]
        d0, d1, d2, l10, l20, l21 = params[3:]
        L = np.array([[np.exp(d0), 0., 0.], [l10, np.exp(d1), 0.],
                      [l20, l21, np.exp(d2)]])
        return bias, L
    def residual(params):
        b, L = unpack(params)
        corrected = (A - b) @ L.T
        return np.einsum('ij,ij->i', corrected, corrected) - 1.0
    x0 = np.r_[A.mean(axis=0), np.zeros(6)]
    result = least_squares(residual, x0, loss="soft_l1", f_scale=0.05, max_nfev=2000)
    if not result.success:
        raise RuntimeError(f"Accelerometer fit did not converge: {result.message}")
    return unpack(result.x)


def estimate_six_pose_rotation(means: np.ndarray, known_body_directions: np.ndarray) -> np.ndarray:
    """Kabsch alignment of *known, ordered* stationary body directions.

    Unlike choosing the dominant measured axis, supplied labels preserve an
    independently meaningful body frame (e.g. which side is physically +X).
    ``means`` must ALREADY be ellipsoid-corrected and normalized to unit g.
    """
    A = np.asarray(means, float)
    B = np.asarray(known_body_directions, float)
    if A.shape != (6, 3) or B.shape != (6, 3):
        raise ValueError("Expected six vectors in each 6x3 matrix")
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt


def estimate_gyro_bias(still_gyro_dps: np.ndarray) -> np.ndarray:
    """Estimate steady zero-rate offset, preferably after thermal warm-up."""
    data = np.asarray(still_gyro_dps, float)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 10:
        raise ValueError("Need at least 10 motionless gyro samples of shape (N,3)")
    return np.mean(data, axis=0)


def fit_gyro_axis_matrix(axis_rotation_samples: np.ndarray) -> np.ndarray:
    """Approximate gyro axis alignment using three independent rotations.

    ``axis_rotation_samples`` contains rows of mean bias-corrected rates for
    *known* positive body-X, body-Y and body-Z rotations (deg/s). The inverse
    of their normalized column vectors maps raw gyro axes into body axes.
    This estimates axes only, NOT accurate angular-rate scale factors.
    """
    rates = np.asarray(axis_rotation_samples, float)
    if rates.shape != (3, 3):
        raise ValueError("Expected a 3x3 matrix of mean gyro rate vectors")
    norms = np.linalg.norm(rates, axis=1)
    if np.any(norms < 1e-4):
        raise ValueError("Rotations must have measurable angular speed")
    basis = (rates / norms[:, None]).T
    if abs(np.linalg.det(basis)) < 1e-4:
        raise ValueError("Rotation axes are linearly dependent")
    return np.linalg.inv(basis)
