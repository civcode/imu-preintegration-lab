"""Hardware-independent regression tests for the *sequential* cube experiment."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from imu_lab.calibration import Calibration
from imu_lab.staged_calibration import (
    AXES, FACES, STAGES, StageStore, fit_cube_faces, fit_signed_gyro_rotations,
    fit_stationary_gyro, load_raw_stage, record_raw_stage, rotation_mask,
)
from imu_lab.ftdi_mpu6050 import MPU6050Ftdi
from test_ftdi import FakePort, FakeController


def capture_fixture(path, a, w, *, stop_reason='completed', overflow=False, sample_div=4):
    """Write synthetic raw g/degree/s stage recording in exactly the on-disk format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('t_s', 'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'))
        for i, (acc, gyro) in enumerate(zip(a, w)):
            writer.writerow([i * (1 + sample_div) / 1000., *acc, *gyro])
    path.with_suffix('.metadata.json').write_text(json.dumps({
        'sample_div': sample_div, 'dlpf_cfg': 3,
        'sample_rate_hz': 1000. / (1 + sample_div),
        'stop_reason': stop_reason, 'overflow_detected': overflow,
        'samples': len(a)}))
    return path


def test_raw_stage_capture_interrupt_saves_complete_packets_and_closes(tmp_path):
    port = FakePort(packet_count=12, interrupt_after=4)
    ctrl = FakeController(port)
    result = record_raw_stage(tmp_path / 'trial.csv', seconds=3,
                              verbose=False, controller_factory=lambda: ctrl)
    assert result.stop_reason == 'interrupted'
    assert ctrl.terminated
    assert len(result.samples) == 4
    with pytest.raises(ValueError, match='Ctrl\\+C'):
        load_raw_stage(tmp_path / 'trial.csv')
    t, a, w = load_raw_stage(tmp_path / 'trial.csv', allow_interrupted=True)
    np.testing.assert_allclose(t, np.arange(4) / 200)
    np.testing.assert_allclose(a, np.tile([0, 0, 1], (4, 1)))


def test_raw_capture_overflow_rejected_and_config_mismatch(tmp_path):
    file = capture_fixture(tmp_path / 'overflow.csv', np.zeros((30, 3)),
                           np.zeros((30, 3)), overflow=True)
    with pytest.raises(ValueError, match='overflow'):
        load_raw_stage(file)
    file = capture_fixture(tmp_path / 'valid.csv', np.zeros((30, 3)),
                           np.zeros((30, 3)), sample_div=9)
    with pytest.raises(ValueError, match='Sensor settings differ'):
        load_raw_stage(file)


def test_six_pose_fit_uses_ellipsoid_output_and_explicit_physical_labels():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(123)
    Rtrue = Rotation.from_euler('xyz', [3, -4, 5], degrees=True).as_matrix()
    L = np.array([[1.03, 0, 0], [.015, .98, 0], [-.01, .02, 1.04]])
    b = np.array([.04, -.025, .018])
    raw = {}
    for f in FACES:
        body = np.tile(AXES[f], (250, 1))
        raw[f] = ((body @ Rtrue @ np.linalg.inv(L).T) + b +
                  rng.normal(0, .002, (250, 3)),
                  rng.normal(0, .012, (250, 3)))
    R, info = fit_cube_faces(raw, b, L, max_rate_dps=.2, min_samples=100)
    np.testing.assert_allclose(R, Rtrue, atol=.006)
    assert max(info['angular_residual_deg'].values()) < .35
    assert all(n >= 100 for n in info['counts'].values())


def test_gyro_bias_estimated_only_after_corrected_gravity_check():
    rng = np.random.default_rng(11)
    b = [.12, -.03, .20]
    a = rng.normal([.1, 0, 1], .004, (500, 3))
    a[0] = [0, 0, 0]  # obvious invalid accel sample should be dropped
    w = rng.normal(b, .012, (500, 3))
    cal = Calibration.identity()
    cal.accel_bias_g = np.array([.1, 0, 0])
    got, report = fit_stationary_gyro(a, w, cal, max_gravity_error_g=.05)
    np.testing.assert_allclose(got, b, atol=.003)
    assert report['rejected'] >= 1


def test_signed_axis_rotation_selection_and_recovery_from_bias_corrected_data():
    rng = np.random.default_rng(7)
    # columns are true axis response directions, similar to sensor cross-axis error
    U = np.array([[1, .025, -.01], [.018, 1, -.02], [-.02, .03, 1]])
    U = U / np.linalg.norm(U, axis=0)
    measured = {}
    for f in FACES:
        speed = 125.0
        measured[f] = np.tile(speed * (U @ AXES[f]), (220, 1)) + rng.normal(0, .6, (220, 3))
    T, report = fit_signed_gyro_rotations(measured, min_dps=90, max_dps=180,
        direction_cosine=.95, max_subordinate_dps=12, min_samples=100)
    np.testing.assert_allclose(T @ U, np.eye(3), atol=.003)
    assert all(n > 100 for n in report['selected_counts'].values())
    assert max(report['positive_negative_angle_deg'].values()) < .5
    # A threshold borrowed blindly from ±250 °/s experiments could reject all.
    with pytest.raises(ValueError, match='selected rotations'):
        fit_signed_gyro_rotations(measured, min_dps=200, max_dps=240,
          direction_cosine=.95, max_subordinate_dps=12, min_samples=100)
    with pytest.raises(ValueError, match='full scale'):
        rotation_mask(measured['+X'], '+X', min_dps=200, max_dps=300,
                      direction_cosine=.9, max_subordinate_dps=10)


def test_stage_checkpoints_enforce_order_and_invalidate_downstream(tmp_path):
    store = StageStore(tmp_path / 'calibration')
    raw = capture_fixture(tmp_path / 'raw.csv', np.tile([0,0,1.], (100,1)),
                           np.zeros((100,3)))
    with pytest.raises(FileNotFoundError):
        store.save(STAGES[1], {'R': np.eye(3).tolist()}, raw_files=[raw], settings={})
    store.save(STAGES[0], {'bias_g':[0,0,0], 'L':np.eye(3).tolist()},
               raw_files=[raw], settings={'cutoff':3})
    store.save(STAGES[1], {'R': np.eye(3).tolist()}, raw_files=[raw], settings={})
    store.save(STAGES[2], {'bias_dps':[0,0,0]}, raw_files=[raw], settings={})
    store.save(STAGES[3], {'T':np.eye(3).tolist()}, raw_files=[raw], settings={})
    np.testing.assert_allclose(store.final_calibration().gyro_matrix, np.eye(3))
    store.save(STAGES[0], {'bias_g':[.1,0,0], 'L':np.eye(3).tolist()},
               raw_files=[raw], settings={'cutoff':5}, overwrite=True)
    with pytest.raises(ValueError, match='stale'):
        store.final_calibration()
    with pytest.raises(ValueError, match='stale'):
        store.save(STAGES[2], {'bias_dps':[0,0,0]}, raw_files=[raw], settings={}, overwrite=True)
    store.save(STAGES[1], {'R':np.eye(3).tolist()}, raw_files=[raw],
               settings={}, overwrite=True)
    # Changing the physical recording invalidates even the first checkpoint.
    raw.write_text(raw.read_text()+'\n')
    with pytest.raises(ValueError, match='original data changed'):
        store.load(STAGES[0])


def test_notebook_calibration_is_guided_and_cells_interrupt_safe():
    import nbformat
    from pathlib import Path
    note = nbformat.read(Path(__file__).resolve().parents[1] / 'notebooks/01_calibration.ipynb', as_version=4)
    all_text = '\n'.join(''.join(c.source) for c in note.cells)
    for label in ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5',
                  'ACCEPT_ELLIPSOID', 'ACCEPT_CUBE_ALIGNMENT',
                  'ACCEPT_GYRO_BIAS', 'ACCEPT_GYRO_AXES',
                  'ACCEPT_FINAL_CALIBRATION']:
        assert label in all_text
    for cell in note.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '01_calibration.ipynb', 'exec')
            assert 'except KeyboardInterrupt:' in cell.source
