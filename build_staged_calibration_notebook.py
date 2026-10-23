"""Regenerate the guided cube-calibration notebook from readable source cells.

Run manually: python build_staged_calibration_notebook.py
This file only writes a notebook; it never accesses sensor hardware.
"""
from pathlib import Path
import nbformat as nb

NOTEBOOK = Path(__file__).resolve().parent / "notebooks" / "01_calibration.ipynb"
n = nb.v4.new_notebook()
cells = []

def md(s):
    cells.append(nb.v4.new_markdown_cell(s.strip()))

def code(s):
    # All user-running cells gracefully handle notebook Kernel > Interrupt
    # without leaving a live FTDI connection. Context-manager capture also
    # handles interrupt during an active sensor read.
    body = '\n'.join('    '+line if line else '' for line in s.strip().split('\n'))
    cells.append(nb.v4.new_code_cell('try:\n'+body+'\nexcept KeyboardInterrupt:\n'
         '    print("\\nInterrupted safely; previous recordings/checkpoints remain on disk.\")'))

md(r"""
# 01 — Hands-on *sequential* cube calibration (FT232H / MPU-6050)

This is a **physical experiment**, not a one-shot calibration function. Your sensor must be mounted **rigidly** in an orthogonal physical cube. You deliberately vary the placement and rotation direction, inspect data quality, tune selection gates, and **accept each stage before the next stage can be computed**.

**Do not run all notebook cells blindly.** Execute one stage at a time, follow its physical instructions, inspect its plots, edit the selection thresholds, then change its `ACCEPT_...` switch to `True` only if satisfied.

**Calibration chain** (original notebook `ftdi-dt232h-mpu6050-05.ipynb`, reorganized):

1. Multi-orientation **stationary accelerometer data → ellipsoid bias/scale/cross-axis matrix** `b, L`.
2. Using `L, b`, six labeled **cube faces → accelerometer-to-cube alignment** `R` (not a world attitude estimate).
3. With corrected accelerometer available to check stillness, **stationary gyroscope → offset** `b_g`.
4. With `b_g` applied, **±X/±Y/±Z controlled rotations → gyro internal-axis mapping** `T`. *Choose rates and thresholds from the real data and the configured ±250°/s full scale.*
5. Collect **new, independent stationary data**, check calibrated measurements and spurious drift, and *only then* export `data/calibration.json`.

The cube defines our common **body axes**. In a face experiment, `+X` means the cube's **+X direction points upward**, so the stationary specific-force vector in cube coordinates is +X. For a signed rotation, `+X` means rotate according to the right-hand rule about the cube's positive X axis. Record the corresponding negative turns, too. *Mark the three + axes physically on the cube.*

Accelerometer units in raw recordings: **g**. Gyro: **degrees/second**. The recorder configures accelerometer ±2g and gyro ±250°/s, DLPF=3, sample divisor=4 → nominal 200 Hz. Every capture saves raw CSV plus metadata, timestamps by nominal FIFO sample interval, stops on overflow, and safely closes FTDI on Ctrl+C. Interrupted partial files are saved for inspection but **not accepted without explicit opt-in**.

No physical hardware is contacted until you execute a recording cell. Notebook `04_preintegration.ipynb` contains a separate hardware-free simulation.
""")
code(r"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root = Path.cwd().resolve()
if not (root / 'imu_lab').is_dir():
    root = root.parent  # when Jupyter starts in notebooks/
if not (root / 'imu_lab').is_dir():
    raise RuntimeError('Start JupyterLab from the project root or notebooks directory')
import sys
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from imu_lab.calibration import Calibration, fit_accelerometer_ellipsoid
from imu_lab.staged_calibration import (
    FACES, STAGES, AXES, StageStore, record_raw_stage, load_raw_stage,
    still_mask, fit_cube_faces, fit_stationary_gyro,
    fit_signed_gyro_rotations, rotation_mask,
)
from imu_lab.core import GRAVITY, ImuData, estimate_initial_state, integrate_direct

DATA = root / 'data' / 'calibration_stages'
DATA.mkdir(parents=True, exist_ok=True)
store = StageStore(DATA)
# Keep these constant across ALL experiments; full scales are fixed in the recorder.
FTDI_URL = 'ftdi://ftdi:232h/1'
SAMPLE_DIV = 4  # 1000/(1+4) = 200 Hz (with DLPF_CFG=3)
DLPF_CFG = 3
ALLOW_INTERRUPTED = False  # deliberately set True only after reviewing partial recording
print('Raw captures/checkpoints:', DATA)
print('Sensor full-scale: accel ±2g, gyro ±250°/s. Nominal:', 1000/(1+SAMPLE_DIV), 'Hz')
""")
md(r"""
## Stage 1 — Collect many *stationary* cube orientations (accelerometer ellipsoid)

**Physical experiment**: Mount the MPU-6050 firmly to a cube. Move it between many distinct faces, edges and corners, **pausing motionless for 1–2 seconds** in every orientation. Cover the full 3D sphere; repeatedly turning about just one axis produces a poor fit. Start with 90 seconds of recording; shorter runs are acceptable if they have sufficient diverse stationary poses. Avoid impacts and clipping at ±2g.

This capture is **raw**: stage-1 data must not have any preceding calibration applied. The recording is saved to `01_multi_orientation.csv`; Ctrl+C safely closes USB and preserves the samples collected so far. Interrupted data require `ALLOW_INTERRUPTED=True` when loading.
""")
code(r"""
raw1 = DATA / '01_multi_orientation.csv'
if raw1.exists():
    print('Existing capture:', raw1, '- reusing; delete/rename it to acquire again')
else:
    input('Prepare cube, then press Enter to BEGIN stage-1 orientation experiment...')
    result = record_raw_stage(raw1, seconds=90, url=FTDI_URL,
                               sample_div=SAMPLE_DIV, dlpf_cfg=DLPF_CFG)
    print('Saved', len(result.samples), 'raw samples. Stop reason:', result.stop_reason)
""")
md(r"""
### Explore the stage-1 static selection — **experiment with the rate gate**

Your original notebook started with `||gyro|| < 3.7 °/s`. That is only a starting point. **Gyro bias has not been calibrated yet**, so we initially use the raw gyro norm. Examine its histogram, then choose `ACC_STATIC_DPS` to retain stationary portions while rejecting most motion. Inspect 3D spread and the number of retained samples. A large sample count concentrated in only one pose is **not** good orientation coverage.
""")
code(r"""
t1, a1, w1 = load_raw_stage(raw1, allow_interrupted=ALLOW_INTERRUPTED,
                             sample_div=SAMPLE_DIV, dlpf_cfg=DLPF_CFG)
ACC_STATIC_DPS = 3.7       # EDIT after inspecting the observed histogram
STATIC_SUBSAMPLE = 3      # optimization speed only; retain full raw recording
mask1 = still_mask(w1, ACC_STATIC_DPS)
print(f'Stage 1: {mask1.sum()} / {len(mask1)} stationary candidates')
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(np.linalg.norm(w1,axis=1), bins=100)
ax[0].axvline(ACC_STATIC_DPS, color='tab:red', label='selected cutoff')
ax[0].set(xlabel='Raw gyro norm (°/s)', ylabel='Samples', title='Experiment with this threshold')
ax[0].legend()
selected1 = a1[mask1][::STATIC_SUBSAMPLE]
if len(selected1) < 100:
    raise ValueError('Too few still samples. Change threshold, or recapture.')
ax[1].scatter(selected1[:,0], selected1[:,1], s=1, alpha=.4)
ax[1].set(xlabel='Raw ax (g)', ylabel='Raw ay (g)', title='XY projection of selected poses')
ax[1].axis('equal')
plt.tight_layout(); plt.show()
""")
code(r"""
bias1, L1 = fit_accelerometer_ellipsoid(selected1)
cal1 = (selected1 - bias1) @ L1.T
radius1 = np.linalg.norm(cal1, axis=1)
print('Accelerometer bias [g]:', bias1)
print('Lower-triangular correction L:\n', L1)
print('Corrected |a| [g]: mean', radius1.mean(), 'RMS error', np.sqrt(np.mean((radius1-1)**2)))
fig, ax = plt.subplots(1,2,figsize=(11,3))
ax[0].hist(np.linalg.norm(selected1,axis=1),bins=60); ax[0].set_title('Before ellipsoid fit: |a|')
ax[1].hist(radius1,bins=60); ax[1].set_title('After fit: |L(a-b)| ≈ 1g')
plt.tight_layout();plt.show()
""")
md('**Acceptance checkpoint.** Check rotation diversity, fitted matrix condition, corrected gravity-norm spread and data quality. Change the static cutoff or recapture if necessary. Downstream stages are tied to this exact result and recording.')
code(r"""
ACCEPT_ELLIPSOID = False  # <-- explicitly change when satisfied
if ACCEPT_ELLIPSOID:
    path = store.save(STAGES[0], {'bias_g': bias1.tolist(), 'L': L1.tolist()},
                      raw_files=[raw1],
                      settings={'static_max_dps': ACC_STATIC_DPS, 'subsample': STATIC_SUBSAMPLE,
                                'selected': int(mask1.sum())}, overwrite=True)
    print('Accepted stage 1:', path)
else:
    print('NOT ACCEPTED. Tune static selection and re-run the fit/plots first.')
""")
md(r"""
## Stage 2 — Six **separately labeled** stationary cube faces

Stage 1's `L` and `b` **must be accepted first**. Set down the cube so that the indicated *positive/negative cube axis points vertically upward*. Each pose should remain motionless for ~4 s. The original notebook grouped poses by direction cosine (>0.9) after ellipsoid calibration; recording each explicitly labeled pose avoids accidentally swapping the physical ±X/±Y/±Z directions. Here we retain the same six-direction Kabsch fit but use the labels as ground truth.

**Important:** This produces *accelerometer-to-cube/body alignment*, not an estimate of a moving sensor's world orientation. No magnetometer/yaw is inferred.
""")
code(r"""
e = store.load(STAGES[0])  # raises if stage 1 has not been accepted
bias1, L1 = np.asarray(e['bias_g']), np.asarray(e['L'])
# Every prompt is a separate physical placement; completed poses survive Ctrl+C.
for face in FACES:
    dest = DATA / f'02_face_{face.replace("+","pos").replace("-","neg")}.csv'
    if dest.exists():
        print(face, ': existing recording; skipping')
        continue
    input(f'Place {face} CUBE axis straight UP, hold still, then press Enter...')
    capture = record_raw_stage(dest, seconds=4, url=FTDI_URL,
                               sample_div=SAMPLE_DIV, dlpf_cfg=DLPF_CFG)
    print(face, capture.stop_reason, len(capture.samples), 'samples')
    if capture.stop_reason != 'completed':
        print('Stopped early: review the partial file before continuing')
        break
""")
md('### Fit six-face alignment **after applying the ellipsoid**. Tune stillness cutoff and inspect the angular error of every labeled pose.')
code(r"""
FACE_STILL_DPS = 3.7
FACE_MIN_SAMPLES = 100
face_files = [DATA / f'02_face_{f.replace("+","pos").replace("-","neg")}.csv' for f in FACES]
face_raw = {}
for f, path in zip(FACES, face_files):
    t, a, w = load_raw_stage(path, allow_interrupted=ALLOW_INTERRUPTED,
                             sample_div=SAMPLE_DIV, dlpf_cfg=DLPF_CFG)
    face_raw[f] = (a, w)
R2, face_report = fit_cube_faces(face_raw, bias1, L1, FACE_STILL_DPS,
                                  min_samples=FACE_MIN_SAMPLES)
print('Sensor -> cube rotation:\n', R2)
for f in FACES:
    print(f, 'still samples:', face_report['counts'][f],
          'error:', round(face_report['angular_residual_deg'][f],3),'deg',
          '|a|:', round(face_report['gravity_norm_g'][f],4), 'g')
fig, ax = plt.subplots(figsize=(8,3))
ax.bar(FACES, [face_report['angular_residual_deg'][f] for f in FACES]);
ax.set(ylabel='Angular residual (°)',title='Error relative to physical cube axes');plt.show()
""")
code(r"""
ACCEPT_CUBE_ALIGNMENT = False  # <-- verify all six labels and residuals
if ACCEPT_CUBE_ALIGNMENT:
    print('Accepted:',store.save(STAGES[1], {'R': R2.tolist(), 'report': face_report},
        raw_files=face_files, settings={'still_max_dps': FACE_STILL_DPS,
                                      'min_samples': FACE_MIN_SAMPLES},overwrite=True))
else:
    print('NOT ACCEPTED. Correct face placement, thresholds or stage 1 first.')
""")
md(r"""
## Stage 3 — Gyroscope zero-rate offset, **using earlier accel calibration**

The cube is placed **completely stationary**, preferably after thermal warm-up. Bias can change with temperature. This raw capture is checked with the **already corrected accelerometer gravity norm** before computing gyro bias. Look at fluctuations to verify you are truly still; do not use the earlier axis rotations for zero-rate estimation.
""")
code(r"""
store.load(STAGES[1])
raw3 = DATA / '03_gyro_stationary.csv'
if not raw3.exists():
    input('Leave cube motionless and press Enter for a 20-second gyro-bias capture...')
    cap = record_raw_stage(raw3, seconds=20, url=FTDI_URL,
                            sample_div=SAMPLE_DIV, dlpf_cfg=DLPF_CFG)
    print('Saved',len(cap.samples),'samples:',cap.stop_reason)
else:
    print('Reusing existing:',raw3)
""")
code(r"""
previous = Calibration(np.asarray(store.load(STAGES[0])['bias_g']),
                       np.asarray(store.load(STAGES[0])['L']),
                       np.asarray(store.load(STAGES[1])['R']),
                       np.zeros(3), np.eye(3))
t3, a3, w3 = load_raw_stage(raw3,allow_interrupted=ALLOW_INTERRUPTED,
                             sample_div=SAMPLE_DIV,dlpf_cfg=DLPF_CFG)
MAX_GRAVITY_ERROR_G = 0.15  # EDIT after inspecting accelerometer stability
MAX_STILL_GYRO_STD_DPS = 1.0 # EDIT after inspecting per-axis rate fluctuations
bias3, bias_report = fit_stationary_gyro(a3,w3,previous,
                                          max_gravity_error_g=MAX_GRAVITY_ERROR_G,
                                          max_gyro_std_dps=MAX_STILL_GYRO_STD_DPS)
print('Gyro bias (°/s):',bias3,'\nDiagnostics:',bias_report)
fig,axs=plt.subplots(2,1,figsize=(11,5))
axs[0].plot(t3,w3);axs[0].set(ylabel='Raw gyro (°/s)',title='A stationary rate should be nearly constant')
a3_corrected = (previous.accel_body_rotation @
                (previous.accel_matrix @ (a3-previous.accel_bias_g).T)).T
axs[1].plot(t3,np.linalg.norm(a3_corrected,axis=1));axs[1].axhline(1,color='black')
axs[1].set(xlabel='Sensor time (s)',ylabel='Corrected |a| (g)');plt.tight_layout();plt.show()
""")
code(r"""
ACCEPT_GYRO_BIAS = False
if ACCEPT_GYRO_BIAS:
    print('Accepted:',store.save(STAGES[2],{'bias_dps':bias3.tolist(),'report':bias_report},
          raw_files=[raw3],settings={'max_gravity_error_g':MAX_GRAVITY_ERROR_G,
                                                 'max_gyro_std_dps':MAX_STILL_GYRO_STD_DPS},overwrite=True))
else:
    print('NOT ACCEPTED. Inspect motion, vibration, temperature and drift.')
""")
md(r"""
## Stage 4 — Gyroscope *internal axis misalignment*, six controlled cube turns

**This is NOT an arbitrary spin capture.** Stage 3 bias must be accepted, and you now deliberately rotate the cube about known, orthonormal cube axes: `+X, −X, +Y, −Y, +Z, −Z`. Use an edge/pivot or a rig to constrain the intended rotation plane as in your original experiment. Keep the other angular components as small as possible, while avoiding hard stops and clipping. The sensor's gyro is configured for **±250°/s**, and readings close to the rail are suspect.

The original notebook used `w_min=200`, `w_max=240`, `direction cosine>0.9`, `max subordinate=10°/s`. **Those are properties of that experiment, not universal defaults.** We first record raw samples from each labeled turn, subtract the *measured* gyro bias, visualize dominant and cross-axis magnitudes, and THEN tune all four gates against actual captured distributions. Tight subordinate rejection can conceal the very cross-axis misalignment we want to estimate.

We estimate direction vectors of the measured rates for each physical axis. Reverse the sign of the −axis trials, average with +axis trials, construct `U=[ux,uy,uz]`, then `T=U⁻¹`. **This is axis-alignment calibration, not full gyroscope scale-factor calibration.**
""")
code(r"""
store.load(STAGES[2])
for face in FACES:
    dest = DATA / f'04_turn_{face.replace("+","pos").replace("-","neg")}.csv'
    if dest.exists():
        print(face, ': existing recording; skipping')
        continue
    input(f'Prepare smooth, physically constrained {face} rotation. Press Enter to capture 12 s...')
    cap = record_raw_stage(dest,seconds=12,url=FTDI_URL,
                            sample_div=SAMPLE_DIV,dlpf_cfg=DLPF_CFG)
    print(face,cap.stop_reason,len(cap.samples),'samples')
    if cap.stop_reason != 'completed':
        print('Stopped early: review this file before continuing'); break
""")
md(r"""
### First **inspect the recorded rates**, then set thresholds

The plot shows each *signed* axis' rate after stage-3 bias removal, plus the magnitude of the two subordinate axes. Capture speed and actual misalignment determine useful selection gates; they should not be hard-coded from another trial. The gyro full scale is ±250°/s in this configuration. It is better to pick an uncontaminated, unsaturated band than to maximize sample count.
""")
code(r"""
bias3 = np.asarray(store.load(STAGES[2])['bias_dps'])
turn_files = [DATA / f'04_turn_{f.replace("+","pos").replace("-","neg")}.csv' for f in FACES]
rates = {}
fig, axs = plt.subplots(3,2,figsize=(13,9))
for face, file, ax in zip(FACES,turn_files,axs.ravel()):
    t,a,w = load_raw_stage(file,allow_interrupted=ALLOW_INTERRUPTED,
                            sample_div=SAMPLE_DIV,dlpf_cfg=DLPF_CFG)
    w = w - bias3  # critical: prior stage applied BEFORE selection
    rates[face] = w
    axis_idx = FACES.index(face)//2
    dominant_signed = w @ AXES[face]
    cross = np.linalg.norm(np.delete(w,axis_idx,axis=1),axis=1)
    ax.plot(t,dominant_signed,label='rate in intended signed direction')
    ax.plot(t,cross,label='subordinate-axis magnitude',alpha=.75)
    ax.set(title=f'{face}: peak {dominant_signed.max():.0f}°/s, cross median {np.median(cross):.1f}°/s',
           xlabel='s',ylabel='°/s')
    ax.axhline(250,color='r',ls=':',label='sensor limit')
axs[0,0].legend();plt.tight_layout();plt.show()
""")
md(r"""
### Select a reproducible rate window; examine **how many** samples each gate keeps

Edit and rerun the following two cells as often as needed. The shown values are exploration starters only, *not calibrated MPU-6050 constants*. For each signed physical axis, the samples must satisfy all the separate conditions: (1) correct signed direction, (2) speed window, (3) direction cosine, (4) subordinate component gate. Always inspect ± symmetry and sufficient counts for each of the six runs.
""")
code(r"""
MIN_RATE_DPS = 50.0         # EDIT using your measured rotation speed (source experiment: 200)
MAX_RATE_DPS = 180.0        # EDIT; must be < 250 full scale (source: 240)
DIRECTION_COSINE = 0.90     # EDIT; angular agreement with physical axis
MAX_SUBORDINATE_DPS = 35.  # EDIT; source: 10, potentially too restrictive
MIN_SELECTED_PER_TURN = 20 # at nominal 200 Hz
fig, axs = plt.subplots(3,2,figsize=(13,8))
for face, ax in zip(FACES,axs.ravel()):
    w=rates[face]
    good=rotation_mask(w,face,min_dps=MIN_RATE_DPS,max_dps=MAX_RATE_DPS,
                       direction_cosine=DIRECTION_COSINE,max_subordinate_dps=MAX_SUBORDINATE_DPS)
    d=w@AXES[face]
    ax.hist(d,bins=70,alpha=.6,label=f'all {len(w)}')
    ax.hist(d[good],bins=40,alpha=.7,label=f'selected {good.sum()}')
    ax.axvline(MIN_RATE_DPS,color='red');ax.axvline(MAX_RATE_DPS,color='red')
    ax.set(title=f'{face}: {good.sum()} selected',xlabel='Rate projected on signed cube axis (°/s)')
axs[0,0].legend();plt.tight_layout();plt.show()
""")
code(r"""
T4, axis_report = fit_signed_gyro_rotations(
    rates, min_dps=MIN_RATE_DPS, max_dps=MAX_RATE_DPS,
    direction_cosine=DIRECTION_COSINE,
    max_subordinate_dps=MAX_SUBORDINATE_DPS,
    min_samples=MIN_SELECTED_PER_TURN)
print('Selected per signed trial:',axis_report['selected_counts'])
print('Mean bias-corrected axis responses (°/s):\n',np.asarray(axis_report['mean_rate_vectors_dps']))
print('Estimated body-axis map T = inverse(U):\n',T4)
print('Axis matrix condition:',axis_report['condition_number'])
print('Opposite-direction consistency (deg):',axis_report['positive_negative_angle_deg'])
""")
code(r"""
ACCEPT_GYRO_AXES = False
if ACCEPT_GYRO_AXES:
    print('Accepted:',store.save(STAGES[3],{'T':T4.tolist(),'report':axis_report},
        raw_files=turn_files,settings={
            'min_rate_dps':MIN_RATE_DPS, 'max_rate_dps':MAX_RATE_DPS,
            'direction_cosine':DIRECTION_COSINE,
            'max_subordinate_dps':MAX_SUBORDINATE_DPS,
            'min_samples_per_direction':MIN_SELECTED_PER_TURN},overwrite=True))
else:
    print('NOT ACCEPTED. Inspect speed distributions, cross-axis data and ± consistency.')
""")
md(r"""
## Stage 5 — Independent validation before exporting **real** parameters

Collect **new data**, not any of the calibration captures. First keep the cube motionless for 20 seconds. Evaluate residual angular rate, corrected gravity norm and the *false position drift* produced by direct IMU integration. This does not establish absolute accuracy under motion; it checks for obvious inconsistencies. If poor, revisit the appropriate upstream stage and recompute every affected downstream stage.

Only after inspection, change `ACCEPT_FINAL_CALIBRATION` to `True`. The final file is generated from validated, dependency-matched stage checkpoints. The `data/calibration.json` file is only created by this final acceptance cell.
""")
code(r"""
params = store.final_calibration()  # validates all upstream stages and raw inputs
raw5 = DATA / '05_independent_still.csv'
if not raw5.exists():
    input('Place cube motionless. Press Enter for an independent 20-second validation capture...')
    cap = record_raw_stage(raw5,seconds=20,url=FTDI_URL,
                            sample_div=SAMPLE_DIV,dlpf_cfg=DLPF_CFG)
    print('Saved',len(cap.samples),'samples:',cap.stop_reason)
else:
    print('Reusing independent stationary run:',raw5)
""")
code(r"""
t5,a5,w5=load_raw_stage(raw5,allow_interrupted=ALLOW_INTERRUPTED,
                         sample_div=SAMPLE_DIV,dlpf_cfg=DLPF_CFG)
params=store.final_calibration()  # do not trust earlier cached state
acc5=(params.accel_body_rotation @ (params.accel_matrix @
      (a5-params.accel_bias_g).T)).T
rate5=(params.gyro_matrix @ (w5-params.gyro_bias_dps).T).T
print('Independent corrected mean |a| (g):',np.linalg.norm(acc5,axis=1).mean())
print('Independent corrected mean w (°/s):',rate5.mean(axis=0))
print('Independent rate std (°/s):',rate5.std(axis=0))
if len(t5)<50:
    raise ValueError('Too few samples for validation')
imu5=ImuData(t=t5,acc=acc5*GRAVITY,gyro=np.deg2rad(rate5))
R05, residual_bg = estimate_initial_state(imu5,static_seconds=min(2.,t5[-1]))
track5=integrate_direct(imu5,R0=R05,gyro_bias=residual_bg)
print('Stationary false final displacement [m]:',np.linalg.norm(track5.p[-1]))
fig, ax = plt.subplots(1,3,figsize=(14,3))
ax[0].plot(t5,np.linalg.norm(acc5,axis=1));ax[0].axhline(1,color='black')
ax[0].set(title='Corrected gravity norm',xlabel='s',ylabel='g')
ax[1].plot(t5,rate5);ax[1].set(title='Corrected gyro',xlabel='s',ylabel='°/s')
ax[2].plot(track5.p[:,0],track5.p[:,1]);ax[2].set(title='False XY motion',xlabel='X [m]',ylabel='Y [m]')
plt.tight_layout();plt.show()
""")
code(r"""
ACCEPT_FINAL_CALIBRATION = False  # <-- confirm stage 5 was good, THEN set True
if ACCEPT_FINAL_CALIBRATION:
    params = store.final_calibration()  # refuses stale inputs / stage params
    # Validate independent recording again before writing the final file.
    load_raw_stage(raw5,allow_interrupted=ALLOW_INTERRUPTED,
                   sample_div=SAMPLE_DIV,dlpf_cfg=DLPF_CFG)
    output = root / 'data' / 'calibration.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    params.to_json(output)
    from hashlib import sha256
    import datetime
    provenance = {'validated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'final_stage_sha256':sha256(store.file(STAGES[-1]).read_bytes()).hexdigest(),
                  'independent_stationary_csv_sha256':sha256(raw5.read_bytes()).hexdigest(),
                  'sample_div':SAMPLE_DIV,'dlpf_cfg':DLPF_CFG,
                  'note':'Refit downstream stages and revalidate if a prior checkpoint changes.'}
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Real calibration saved:',output)
else:
    print('NOT EXPORTED. Adjust/recapture until an independent validation passes.')
""")
md(r"""
### After completion

Proceed to `02_ftdi_record.ipynb` (calibrated acquisition), `03_stationary_validation.ipynb`, then `04_preintegration.ipynb`. For motion capture, the accelerometer and gyroscope must share the cube/body frame, and the JSON parameters must be **applied exactly once**. For another MPU board or a different accelerometer/gyro full scale, redo the physical experiments. The stage files and raw captures provide an audit trail; never use an identity/demo calibration as a real one.

**Limitations:** The MPU-6050 FIFO has no hardware sample timestamps; nominal sensor-clock drift is unmeasured. Cube-axis turns alone cannot determine the full gyro scale-factor matrix. Results depend on mounting rigidity, rotation repeatability, temperature and the intentionally operator-selected gates. If you change an early-stage calibration, the stage store rejects stale downstream checkpoints; rerun each dependent fit rather than mixing inconsistent results.
""")
n['cells']=cells
n['metadata']={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
 "language_info":{"name":"python"}}
NOTEBOOK.parent.mkdir(parents=True,exist_ok=True)
if __name__ == '__main__':
    try:
        nb.write(n, NOTEBOOK)
        print('Generated', NOTEBOOK, 'with', len(cells), 'cells')
    except KeyboardInterrupt:
        print('\nNotebook generation interrupted safely.')
        raise SystemExit(130)
