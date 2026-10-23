"""Known-truth rectangular loop and simulated imperfect IMU for lab checks."""
from dataclasses import dataclass
import numpy as np
from .core import GRAVITY, ImuData


@dataclass
class SyntheticTruth:
    p: np.ndarray
    v: np.ndarray
    a_world: np.ndarray


def make_synthetic(
    sample_rate_hz: int = 100,
    seed: int = 7,
    accel_bias=(0.025, -0.018, 0.010),
    gyro_bias=(0.0007, -0.0005, 0.001),
    accel_noise_std: float = 0.03,
    gyro_noise_std: float = 0.0015,
) -> tuple[ImuData, SyntheticTruth]:
    """A 1m x 1m XY square with pauses, beginning stationary for 2 s.

    Starts level, +Z upward and NO rotations. These are intentionally simple
    known-truth test signals, not a realistic human-held IMU model.
    """
    dt = 1.0 / sample_rate_hz
    t = np.arange(0., 15. + .5 * dt, dt)
    n = len(t)
    a = np.zeros((n, 3))
    def set_acc(lo, hi, axis, value):
        mask = (t >= lo - dt / 4) & (t < hi - dt / 4)
        a[mask, axis] = value
    for lo, hi, axis, value in [
        (2, 3, 0, 1.), (3, 4, 0, -1.),
        (5, 6, 1, 1.), (6, 7, 1, -1.),
        (8, 9, 0, -1.), (9, 10, 0, 1.),
        (11, 12, 1, -1.), (12, 13, 1, 1.),
    ]:
        set_acc(lo, hi, axis, value)
    p, v = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(n - 1):
        p[k + 1] = p[k] + v[k] * dt + 0.5 * a[k] * dt**2
        v[k + 1] = v[k] + a[k] * dt

    rng = np.random.default_rng(seed)
    measured_acc = a + np.array([0., 0., GRAVITY]) + np.asarray(accel_bias)
    measured_acc += rng.normal(0, accel_noise_std, measured_acc.shape)
    measured_gyro = np.zeros((n, 3)) + np.asarray(gyro_bias)
    measured_gyro += rng.normal(0, gyro_noise_std, measured_gyro.shape)
    return ImuData(t, measured_acc, measured_gyro), SyntheticTruth(p, v, a)
