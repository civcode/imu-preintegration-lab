"""Rebuild the educational notebooks. Safe to interrupt with Ctrl+C."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent / 'notebooks'
ROOT.mkdir(exist_ok=True)


def normalized(s):
    # Triple-quoted cells start with an unindented first line followed by
    # indented continuation lines; dedent those continuations independently.
    lines = s.strip("\n").splitlines()
    if len(lines) == 1:
        return lines[0].strip()
    return lines[0].strip() + "\n" + textwrap.dedent("\n".join(lines[1:])).rstrip()


def md(s):
    return nbf.v4.new_markdown_cell(normalized(s))


def code(s):
    """Every executable tutorial cell handles Ctrl+C without killing the kernel."""
    source = normalized(s)
    return nbf.v4.new_code_cell('try:\n' + textwrap.indent(source, '    ') +
                                '\nexcept KeyboardInterrupt:\n    print("Cell interrupted safely; kernel stays usable.")')


def save(name, *cells):
    nb = nbf.v4.new_notebook(cells=list(cells), metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    nbf.write(nb, ROOT / name)


def main():
    save('01_calibration.ipynb',
         md('''# 01 — Calibrating the FT232H + MPU-6050

         This clean teaching notebook reconstructs your original calibration concepts:
         **ellipsoid accelerometer calibration**, **six-pose body alignment**,
         **stationary gyro bias**, and optional **gyro-axis alignment**.

         The synthetic example runs without hardware. For real calibration, set
         `RAW_STATIC_ORIENTATIONS` to a CSV with stationary samples from many
         different orientations, captured using `record_ftdi.py` *without* `--calibration`.
         The CSV contains SI units, so convert the accelerometer back to g before fitting.
         For gyro bias, record a separate 20 s sensor-still CSV. A real six-pose
         fit requires *labeled* +X/-X/+Y/-Y/+Z/-Z resting poses, not unlabeled data.
         All cells catch Ctrl+C. No FTDI connection is held open here.'''),
         code('''from pathlib import Path
         import numpy as np
         import pandas as pd
         from imu_lab.calibration import (Calibration, fit_accelerometer_ellipsoid,
             estimate_six_pose_rotation, estimate_gyro_bias, fit_gyro_axis_matrix)
         from imu_lab.core import GRAVITY
         RAW_STATIC_ORIENTATIONS = None  # e.g. Path('data/many_static_orientations.csv')
         RAW_STILL_GYRO = None  # e.g. Path('data/gyro_still.csv')
         rng = np.random.default_rng(12)
         if RAW_STATIC_ORIENTATIONS is None:
             # DEMO: many known resting orientations with synthetic scale and bias.
             unit = rng.normal(size=(800, 3))
             unit /= np.linalg.norm(unit, axis=1, keepdims=True)
             acc_g = (unit / [1.02, .98, 1.03]) + [.06, -.04, .02]
             print('Using SYNTHETIC calibration readings (not from your hardware).')
         else:
             df = pd.read_csv(RAW_STATIC_ORIENTATIONS)
             acc_g = df[['ax','ay','az']].to_numpy() / GRAVITY
         print('Static orientation samples:', len(acc_g))'''),
         code('''bias, L = fit_accelerometer_ellipsoid(acc_g)
         corrected = (acc_g - bias) @ L.T
         print('Accelerometer bias [g]:', bias)
         print('Triangular calibration L:\\n', L)
         print('Mean |a| before correction [g]:', np.mean(np.linalg.norm(acc_g, axis=1)))
         print('Mean |a| after correction [g]:', np.mean(np.linalg.norm(corrected, axis=1)))'''),
         md('''## Alignment to a chosen body coordinate frame

         A fitted ellipsoid becomes spherical, but this **does not identify sensor
         mounting orientation**. Place the device in six *labeled* positions and
         supply each stationary mean (already corrected by `L`). The vectors
         below are synthetic identity-aligned placeholders. Replace the six rows
         with actual means from your six labeled poses. The ordering is explicitly
         `[+X,-X,+Y,-Y,+Z,-Z]` for both arrays.'''),
         code('''body_dirs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
         # DEMO ONLY. Replace with measured six means after ellipsoid correction.
         measured_pose_means = body_dirs.copy()
         R_body = estimate_six_pose_rotation(measured_pose_means, body_dirs)
         print('Body alignment:\\n', R_body)'''),
         md('''## Gyroscope bias and optional axis mapping

         The sensor must remain motionless for the entire gyro-bias capture.
         If no stationary file is available, the demo uses a known bias. For gyro
         axis mapping, perform separate rotations about each **known positive**
         body axis. The original notebook used rate-based vector alignment;
         this teaches **axis alignment**, not accurate gyro scale calibration.'''),
         code('''if RAW_STILL_GYRO is None:
             still_dps = rng.normal([.12, -.05, .20], .015, size=(1000, 3))
         else:
             still_dps = np.rad2deg(pd.read_csv(RAW_STILL_GYRO)[['gx','gy','gz']].to_numpy())
         gyro_bias = estimate_gyro_bias(still_dps)
         # Replace rows with positive body X/Y/Z mean rotation rates after subtracting bias.
         gyro_axis_means = np.eye(3) * 100.0  # DEMO ONLY
         gyro_matrix = fit_gyro_axis_matrix(gyro_axis_means)
         params = Calibration(bias, L, R_body, gyro_bias, gyro_matrix).validate()
         print('Estimated gyro bias [degrees/s]:', gyro_bias)
         print('Gyro axes -> body transform:\\n', gyro_matrix)'''),
         md('''## Save a validated parameter set

         This example writes to `data/calibration_demo.json`. **Do not reuse the
         synthetic calibration for real recording.** Once all inputs are real,
         save as `data/calibration.json` and pass to `record_ftdi.py`.
         Your existing notebook's pickle keys map as follows:
         `b → accel_bias_g`, `L → accel_matrix`, `aRws → accel_body_rotation`,
         `w_offset → gyro_bias_dps`, `w_rotation → gyro_matrix`.
         Do **not** unpickle calibration files from untrusted sources.'''),
         code('''output = Path('data/calibration_demo.json')
         output.parent.mkdir(exist_ok=True)
         params.to_json(output)
         print('Saved', output)
         print('Warning: DEMO values must not be used on a real MPU-6050.')'''))

    save('02_ftdi_record.ipynb',
         md('''# 02 — Record the FT232H / MPU-6050 FIFO

         Connect the FT232H to the MPU-6050 using your working I²C wiring and
         PyFtdi driver setup. The standalone CLI is `python record_ftdi.py`.

         **Timing:** at DLPF_CFG=3, `SMPLRT_DIV=4` implies a nominal 200 Hz sensor
         sampling rate. All FIFO samples are assigned `sample_index / 200` seconds.
         Python decoding time is **not** used as sample time. Without a per-sample
         hardware timestamp, oscillator error cannot be recovered.

         **Overflow:** on a latched FIFO overflow we stop the recording and write
         metadata identifying it as invalid for contiguous inertial integration.
         `Ctrl+C` in this cell returns partial samples and closes the USB interface.
         Remove the example calibration if you have not recorded a real one.'''),
         code('''from pathlib import Path
         from imu_lab.calibration import Calibration, calibration_callback
         from imu_lab.ftdi_mpu6050 import MPU6050Ftdi
         from imu_lab.recording import save_capture
         FTDI_URL = 'ftdi://ftdi:232h/1'
         RECORD_SECONDS = 20.0
         CALIBRATION_FILE = Path('data/calibration.json')
         OUT_CSV = Path('data/stationary.csv')
         calibration = Calibration.from_json(CALIBRATION_FILE) if CALIBRATION_FILE.exists() else None
         if calibration is None:
             print('No real calibration found: recording uncalibrated SI measurements.')
         print('Output:', OUT_CSV)'''),
         code('''capture_result = None
         # The `with` block guarantees terminate() even if the kernel is interrupted.
         # The capture method catches Ctrl+C and returns all complete received packets.
         with MPU6050Ftdi(url=FTDI_URL, sample_div=4, dlpf_cfg=3,
                          calibration=calibration_callback if calibration else None,
                          calibration_data=calibration) as sensor:
             capture_result = sensor.capture(duration_s=RECORD_SECONDS)
         if capture_result and capture_result.samples:
             save_capture(capture_result, OUT_CSV, calibrated=calibration is not None)
             print('Saved', len(capture_result.samples), 'complete samples.',
                   'Stop:', capture_result.stop_reason)
         else:
             print('No samples to save.')
         if capture_result and capture_result.overflow_detected:
             print('OVERFLOW! Discard this recording for integration; shorten polling/recording.')'''))

    save('03_stationary_validation.ipynb',
         md('''# 03 — Validate a 20-second motionless recording

         Before attempting translation, learn what a **perfectly stationary** IMU
         reports. Accelerometer norm should be near 9.80665 m/s² and angular rate
         near zero after calibration. The position solution may still drift rapidly.

         This notebook checks sidecar metadata and refuses known FIFO-overflow
         recordings. Run notebook 02 or `python record_ftdi.py --seconds 20` first.
         In the absence of real hardware, use the included synthetic square only
         to explore plotting; it is NOT a stationary test.'''),
         code('''from pathlib import Path
         import matplotlib.pyplot as plt
         from validate_stationary import diagnose
         RECORDING = Path('data/stationary.csv')
         if RECORDING.exists():
             diagnostics = diagnose(RECORDING)
             for name, value in diagnostics.items():
                 if not name.startswith('trajectory_'):
                     print(name, ':', value)
         else:
             diagnostics = None
             print('Record a stationary CSV first using notebook 02 or the CLI.')'''),
         code('''if diagnostics is not None:
             fig, ax = plt.subplots(1, 2, figsize=(12, 4))
             for key, label in [('trajectory_raw', 'No residual gyro bias compensation'),
                                ('trajectory_corrected', 'Initial 2s gyro bias removed')]:
                 p = diagnostics[key].p
                 ax[0].plot(p[:,0], p[:,1], label=label)
                 ax[1].plot(diagnostics[key].t, p[:,2], label=label)
             ax[0].scatter([0], [0], marker='x', color='black', label='True rest')
             ax[0].set(xlabel='X [m]', ylabel='Y [m]', title='Apparent XY travel')
             ax[0].axis('equal')
             ax[1].set(xlabel='Time [s]', ylabel='Estimated Z [m]', title='False vertical motion')
             ax[0].legend(); ax[1].legend(); fig.tight_layout(); plt.show()
         else:
             print('No stationary recording selected; no plots generated.')'''))

    save('04_preintegration.ipynb',
         md('''# 04 — Direct integration versus IMU preintegration

         **Learning objective:** integrate the same calibrated specific force
         and angular velocity in two ways. Direct integration updates pose,
         velocity and position each sample. Preintegration sums motion relative
         to each interval's **starting sensor frame**; gravity is added only when
         combining an interval with a world-frame state.

         Both methods drift: preintegration is a convenient *measurement factor*,
         not a gravity/bias correction or a complete sensor-fusion solution.
         Input CSV must contain calibrated acceleration in m/s² and gyro in rad/s
         in the **same right-handed body axes**. Set `DATA_FILE` to your real run.
         If an overflow was detected, loading fails rather than hiding a gap.'''),
         code('''from pathlib import Path
         import numpy as np
         import matplotlib.pyplot as plt
         from scipy.spatial.transform import Rotation
         from imu_lab import (load_csv, estimate_initial_state,
                              integrate_direct, integrate_preintegrated)
         from imu_lab.recording import verify_capture_metadata
         DATA_FILE = Path('data/synthetic_noisy.csv')
         if not DATA_FILE.exists():
             from create_synthetic_data import main as generate_example_data
             generate_example_data()  # local, deterministic demo only
         # Keep the sensor completely STILL for at least the first 2 s.
         verify_capture_metadata(DATA_FILE)
         imu = load_csv(DATA_FILE)
         R0, remaining_gyro_bias = estimate_initial_state(imu, static_seconds=2.0)
         print('Samples:', len(imu.t), 'duration:', imu.t[-1] - imu.t[0], 's')
         print('Residual gyro bias [rad/s]:', remaining_gyro_bias)'''),
         code('''# Keyframe stride = 20 samples; at 200 Hz this would be 0.1 s.
         # Synthetic demo runs at 100 Hz, so here it is 0.2 s.
         STRIDE = 20
         direct = integrate_direct(imu, R0=R0, gyro_bias=remaining_gyro_bias)
         pre, blocks = integrate_preintegrated(
             imu, every_n_samples=STRIDE, R0=R0, gyro_bias=remaining_gyro_bias)
         inds = [0] + [b.end_idx for b in blocks]
         print('Direct / preintegration max mismatch at keyframes [m]:',
               np.max(np.linalg.norm(pre.p - direct.p[inds], axis=1)))
         if blocks:
             first = blocks[0]
             print('First block duration [s]:', first.duration_s)
             print('delta-v [start body m/s]:', first.delta_v)
             print('delta-p [start body m]:', first.delta_p)'''),
         code('''fig = plt.figure(figsize=(12, 5))
         ax = fig.add_subplot(121)
         ax.plot(direct.p[:,0], direct.p[:,1], label='Direct, every IMU sample')
         ax.plot(pre.p[:,0], pre.p[:,1], 'o', ms=2, label='Preintegrated keyframes')
         ax.set(xlabel='X [m]', ylabel='Y [m]', title='XY estimated trajectory')
         ax.axis('equal'); ax.legend()
         ax3 = fig.add_subplot(122, projection='3d')
         ax3.plot(*direct.p.T, label='Direct')
         ax3.plot(*pre.p.T, 'o', ms=2, label='Keyframes')
         ax3.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]', title='3D path')
         fig.tight_layout(); plt.show()'''),
         md('''### Suggested experiments

         1. Change `STRIDE` to 1, 10, 50, 100 and compare endpoint agreement.
         2. Switch to `data/synthetic_ideal.csv` (same known square path).
         3. Record **20 seconds still** and inspect drift before trying movement.
         4. Move the sensor about 30 cm without rotating and return to rest.
         5. Rotate in place: ideally translation remains zero; in practice attitude
            and axis misalignment cause false horizontal acceleration.

         With no magnetometer or external reference, yaw is not observable from a
         stationary accelerometer. Initial yaw is assigned arbitrarily to zero.
         See `imu_lab/core.py` for equations and frame conventions.'''))



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nNotebook generation interrupted. Rerun to finish safely.")
        raise SystemExit(130)
