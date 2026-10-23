import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from imu_lab import (
    GRAVITY, ImuData, estimate_initial_state, integrate_direct,
    integrate_preintegrated, make_synthetic, preintegrate_interval,
)


def test_ideal_square_returns_to_origin():
    imu, truth = make_synthetic(accel_bias=(0, 0, 0), gyro_bias=(0, 0, 0),
                                accel_noise_std=0, gyro_noise_std=0)
    R0, gyro_bias = estimate_initial_state(imu, 2.0)
    traj = integrate_direct(imu, R0=R0, gyro_bias=gyro_bias)
    np.testing.assert_allclose(traj.p, truth.p, rtol=0, atol=1e-9)
    assert np.linalg.norm(traj.p[-1]) < 1e-8
    assert np.max(traj.p[:, 2]) < 1e-9


@pytest.mark.parametrize('stride', [1, 13, 50, 137, 9999])
def test_direct_equals_preintegration_at_keyframes(stride):
    imu, _ = make_synthetic()
    R0, gyro_bias = estimate_initial_state(imu, 2.0)
    direct = integrate_direct(imu, R0, gyro_bias=gyro_bias)
    preint, blocks = integrate_preintegrated(imu, stride, R0, gyro_bias=gyro_bias)
    indices = [0] + [block.end_idx for block in blocks]
    np.testing.assert_allclose(preint.t, imu.t[indices], rtol=0, atol=1e-12)
    np.testing.assert_allclose(preint.p, direct.p[indices], rtol=0, atol=1e-9)
    np.testing.assert_allclose(preint.v, direct.v[indices], rtol=0, atol=1e-9)
    dR = Rotation.from_quat(preint.quat_xyzw).inv() * Rotation.from_quat(direct.quat_xyzw[indices])
    assert max(np.linalg.norm(dR.as_rotvec(), axis=1)) < 1e-9


def test_preintegrated_deltas_are_in_initial_body_frame():
    t = np.arange(0, 3.01, .01)
    # Rotating around world Z while remaining stationary: gravity cancellation.
    imu = ImuData(t, np.tile([0, 0, GRAVITY], (len(t), 1)),
                  np.tile([0, 0, .2], (len(t), 1)))
    direct = integrate_direct(imu)
    preint, _ = integrate_preintegrated(imu, 31)
    assert np.max(np.abs(direct.p)) < 1e-9
    assert np.max(np.abs(preint.p)) < 1e-9
    np.testing.assert_allclose(direct.quat_xyzw[-1], Rotation.from_rotvec([0, 0, .6]).as_quat(), atol=1e-12)
    block = preintegrate_interval(imu, 0, 100)
    assert abs(block.duration_s - 1.0) < 1e-12
    np.testing.assert_allclose(block.delta_R.as_rotvec(), [0, 0, .2], atol=1e-12)


def test_invalid_timestamps_rejected():
    with pytest.raises(ValueError, match='strictly increasing'):
        ImuData([0, .01, .01], np.zeros((3, 3)), np.zeros((3, 3)))


def test_rotating_and_accelerating_reconstructs_same_keyframes():
    t = np.arange(0., 4.001, .01)
    world_linear_acc = np.array([.5, -.2, .1])
    true_R = Rotation.from_rotvec(t[:, None] * [0, 0, .4])
    f_body = true_R.inv().apply(world_linear_acc - [0., 0., -GRAVITY])
    gyro_body = np.tile([0., 0., .4], (len(t), 1))
    imu = ImuData(t, f_body, gyro_body)
    R0 = Rotation.from_euler('xyz', [.2, -.1, .5])
    direct = integrate_direct(imu, R0)
    pre, blocks = integrate_preintegrated(imu, 37, R0)
    ix = [0] + [block.end_idx for block in blocks]
    np.testing.assert_allclose(pre.p, direct.p[ix], atol=1e-10, rtol=0)
    np.testing.assert_allclose(pre.v, direct.v[ix], atol=1e-10, rtol=0)
    dR = Rotation.from_quat(pre.quat_xyzw).inv() * Rotation.from_quat(direct.quat_xyzw[ix])
    assert np.max(np.linalg.norm(dR.as_rotvec(), axis=1)) < 1e-10


def test_initial_stationary_tilt_alignment_cancels_gravity():
    t = np.arange(0., 3.01, .01)
    true_R = Rotation.from_euler('xyz', [.25, -.16, .65])
    body_specific_force = true_R.inv().apply([0., 0., GRAVITY])
    imu = ImuData(t, np.tile(body_specific_force, (len(t), 1)), np.zeros((len(t), 3)))
    R0, gyro_bias = estimate_initial_state(imu, 2.)
    np.testing.assert_allclose(R0.apply(body_specific_force), [0., 0., GRAVITY], atol=1e-10)
    traj = integrate_direct(imu, R0, gyro_bias=gyro_bias)
    assert np.linalg.norm(traj.p[-1]) < 1e-10
