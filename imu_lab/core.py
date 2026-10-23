"""A small, intentionally transparent strapdown INS / IMU preintegration example.

Conventions
-----------
* World frame: right-handed, +Z upward; gravitational acceleration [0, 0, -g].
* R_WB maps vectors from the sensor/body frame B into world coordinates W.
* Gyro values: radians/second, measured in the body frame.
* Accelerometer: specific force in the body frame, m/s^2 (about +g in body
  +Z when motionless and level, NOT zero and NOT linear acceleration).
* Bias vectors are in body axes. Initial velocity and position default to zero.
* Each sample k is treated as constant through [t[k], t[k+1]).
* Angular velocity is integrated on SO(3), using a midpoint orientation to
  rotate specific force into the world frame.

Not included: covariance, noise-density modeling, bias Jacobians/propagation,
optimization, drift correction or ZUPT. This is pedagogical dead reckoning.
"""
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665
G_W = np.array([0.0, 0.0, -GRAVITY])
COLUMNS = ('t_s', 'ax', 'ay', 'az', 'gx', 'gy', 'gz')


@dataclass
class ImuData:
    t: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.acc = np.asarray(self.acc, dtype=float)
        self.gyro = np.asarray(self.gyro, dtype=float)
        n = len(self.t)
        if n < 2 or self.t.ndim != 1 or self.acc.shape != (n, 3) or self.gyro.shape != (n, 3):
            raise ValueError('Expected at least two samples, with t=(N,), acc=(N,3), gyro=(N,3).')
        if not (np.isfinite(self.t).all() and np.isfinite(self.acc).all() and np.isfinite(self.gyro).all()):
            raise ValueError('Non-finite IMU values or timestamps.')
        if np.any(np.diff(self.t) <= 0):
            raise ValueError('Timestamps must be strictly increasing; check dropped/duplicate packets.')


def load_csv(path: str | Path) -> ImuData:
    df = pd.read_csv(path)
    missing = [col for col in COLUMNS if col not in df]
    if missing:
        raise ValueError(f'Missing CSV columns: {missing}; expected {COLUMNS}')
    d = df.loc[:, COLUMNS].to_numpy(dtype=float)
    imu = ImuData(d[:, 0], d[:, 1:4], d[:, 4:7])
    dt = np.diff(imu.t)
    if np.max(dt) > 3 * np.median(dt):
        warnings.warn('Some timestamp gaps exceed 3x the median; inspect dropped samples.', stacklevel=2)
    return imu


def estimate_initial_state(
    imu: ImuData,
    static_seconds: float = 2.0,
    accel_bias=(0., 0., 0.),
) -> tuple[Rotation, np.ndarray]:
    """Initial roll/pitch from gravity direction and gyro bias from stillness.

    Requires the sensor to remain stationary at the beginning of recording.
    A 6-axis IMU cannot determine absolute yaw from gravity, so the chosen
    initial yaw is arbitrary. One stationary pose cannot calibrate full accel
    bias separately from initial tilt; provide prior accel calibration if known.
    """
    static_mask = imu.t < imu.t[0] + static_seconds
    if static_mask.sum() < 10:
        raise ValueError('Need >=10 stationary samples. Increase recording or static_seconds.')
    mean_specific_force = imu.acc[static_mask].mean(axis=0) - np.asarray(accel_bias, dtype=float)
    if abs(np.linalg.norm(mean_specific_force) - GRAVITY) > 1.5:
        warnings.warn('Initial acceleration norm differs significantly from 1g: is the sensor stationary and calibrated?', stacklevel=2)
    if np.linalg.norm(mean_specific_force) < 1e-8:
        raise ValueError('Stationary accelerometer mean is zero; check units.')
    # Find minimal rotation mapping the measured gravity-opposing specific
    # force to world +Z. Rotation about world +Z (absolute heading) is free.
    R0, _ = Rotation.align_vectors(
        np.array([[0., 0., 1.]]), mean_specific_force[None] / np.linalg.norm(mean_specific_force)
    )
    gyro_bias = imu.gyro[static_mask].mean(axis=0)
    return R0, gyro_bias


@dataclass
class Trajectory:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    quat_xyzw: np.ndarray


@dataclass
class PreintegratedInterval:
    start_idx: int
    end_idx: int
    duration_s: float
    delta_R: Rotation
    delta_v: np.ndarray  # In the START body frame
    delta_p: np.ndarray  # In the START body frame


def _biases(accel_bias, gyro_bias):
    b_a = np.asarray(accel_bias, dtype=float)
    b_g = np.asarray(gyro_bias, dtype=float)
    if b_a.shape != (3,) or b_g.shape != (3,):
        raise ValueError('Biases must be length-3 vectors in sensor coordinates.')
    return b_a, b_g


def integrate_direct(
    imu: ImuData,
    R0: Rotation | None = None,
    accel_bias=(0., 0., 0.),
    gyro_bias=(0., 0., 0.),
    initial_velocity=(0., 0., 0.),
) -> Trajectory:
    """Direct 6-DoF inertial dead-reckoning at IMU frequency."""
    b_a, b_g = _biases(accel_bias, gyro_bias)
    n = len(imu.t)
    R = Rotation.identity() if R0 is None else R0
    p, v, quat = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 4))
    v[0] = initial_velocity
    quat[0] = R.as_quat()
    for k in range(n - 1):
        dt = float(imu.t[k + 1] - imu.t[k])
        w = imu.gyro[k] - b_g
        f = imu.acc[k] - b_a
        R_mid = R * Rotation.from_rotvec(0.5 * w * dt)
        a_world = R_mid.apply(f) + G_W
        p[k + 1] = p[k] + v[k] * dt + .5 * a_world * dt**2
        v[k + 1] = v[k] + a_world * dt
        R = R * Rotation.from_rotvec(w * dt)
        quat[k + 1] = R.as_quat()
    return Trajectory(t=imu.t.copy(), p=p, v=v, quat_xyzw=quat)


def preintegrate_interval(
    imu: ImuData,
    start_idx: int,
    end_idx: int,
    accel_bias=(0., 0., 0.),
    gyro_bias=(0., 0., 0.),
) -> PreintegratedInterval:
    """Accumulate Delta R, Delta v, Delta p in start-keyframe BODY coordinates.

    No gravity or world pose appears in the accumulation. This is the core
    mathematical distinction between preintegration and full integration.
    """
    if not (0 <= start_idx < end_idx < len(imu.t)):
        raise ValueError('Require 0 <= start_idx < end_idx < number of samples.')
    b_a, b_g = _biases(accel_bias, gyro_bias)
    dR = Rotation.identity()
    dv, dp = np.zeros(3), np.zeros(3)
    for k in range(start_idx, end_idx):
        dt = float(imu.t[k + 1] - imu.t[k])
        w = imu.gyro[k] - b_g
        f = imu.acc[k] - b_a
        dR_mid = dR * Rotation.from_rotvec(0.5 * w * dt)
        f_start = dR_mid.apply(f)
        dp = dp + dv * dt + 0.5 * f_start * dt**2
        dv = dv + f_start * dt
        dR = dR * Rotation.from_rotvec(w * dt)
    return PreintegratedInterval(
        start_idx=start_idx, end_idx=end_idx,
        duration_s=float(imu.t[end_idx] - imu.t[start_idx]),
        delta_R=dR, delta_v=dv, delta_p=dp,
    )


def integrate_preintegrated(
    imu: ImuData,
    every_n_samples: int = 50,
    R0: Rotation | None = None,
    accel_bias=(0., 0., 0.),
    gyro_bias=(0., 0., 0.),
    initial_velocity=(0., 0., 0.),
) -> tuple[Trajectory, list[PreintegratedInterval]]:
    """Preintegrate each keyframe interval; reconstruct states at boundaries.

    No correction/optimization occurs at boundaries. If implementation is
    correct this gives the same states as direct integration at those indices.
    """
    if every_n_samples <= 0:
        raise ValueError('every_n_samples must be positive')
    keyframes = list(range(0, len(imu.t) - 1, every_n_samples)) + [len(imu.t) - 1]
    m = len(keyframes)
    p, v, quat = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros((m, 4))
    v[0] = initial_velocity
    R = Rotation.identity() if R0 is None else R0
    quat[0] = R.as_quat()
    intervals: list[PreintegratedInterval] = []
    for i in range(m - 1):
        block = preintegrate_interval(imu, keyframes[i], keyframes[i + 1], accel_bias, gyro_bias)
        intervals.append(block)
        dt = block.duration_s
        p[i + 1] = p[i] + v[i] * dt + .5 * G_W * dt**2 + R.apply(block.delta_p)
        v[i + 1] = v[i] + G_W * dt + R.apply(block.delta_v)
        R = R * block.delta_R
        quat[i + 1] = R.as_quat()
    return Trajectory(imu.t[keyframes], p, v, quat), intervals
