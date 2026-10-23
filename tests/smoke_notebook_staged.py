"""Opt-in offline smoke test of all calibration notebook cells, no hardware.

Not part of pytest because it temporarily supplies synthetic capture data,
changes interactive acceptance controls and runs the notebook in a kernel.
"""
import sys
from pathlib import Path
import numpy as np
import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_staged_calibration import capture_fixture
from imu_lab.staged_calibration import AXES, FACES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'calibration_stages'
if DATA.exists():
    raise RuntimeError('Refusing to overwrite existing real calibration data')
DATA.mkdir(parents=True)
rng = np.random.default_rng(42)
b = np.array([.03,-.015,.025])
L = np.array([[1.02,0,0],[.01,.99,0],[-.008,.015,1.015]])
R = np.eye(3)
gb = np.array([.12,-.07,.19])
U = np.array([[1,.03,-.015],[.015,1,-.02],[-.02,.025,1]])
# Stage1: 6 spherical directions and intervening static measurements
v = rng.normal(size=(4500,3)); v = v/np.linalg.norm(v,axis=1)[:,None]
a1 = (v @ np.linalg.inv(L).T) + b + rng.normal(0,.001,(len(v),3))
w1 = gb + rng.normal(0,.02,(len(v),3))
capture_fixture(DATA/'01_multi_orientation.csv',a1,w1)
for face in FACES:
    axes = np.tile(AXES[face],(800,1))
    a = axes @ np.linalg.inv(L).T + b + rng.normal(0,.002,axes.shape)
    w = gb + rng.normal(0,.02,a.shape)
    capture_fixture(DATA/f'02_face_{face.replace("+","pos").replace("-","neg")}.csv',a,w)
a3=np.tile([b[0],b[1],1+b[2]],(2500,1))+rng.normal(0,.002,(2500,3))
w3=gb+rng.normal(0,.02,(2500,3))
capture_fixture(DATA/'03_gyro_stationary.csv',a3,w3)
for face in FACES:
    w=np.tile(120*(U @ AXES[face])+gb,(900,1))+rng.normal(0,.3,(900,3))
    a=np.tile([b[0],b[1],1+b[2]],(900,1))
    capture_fixture(DATA/f'04_turn_{face.replace("+","pos").replace("-","neg")}.csv',a,w)
capture_fixture(DATA/'05_independent_still.csv',a3,w3)

note=nbformat.read(ROOT/'notebooks'/'01_calibration.ipynb',as_version=4)
for c in note.cells:
    if c.cell_type == 'code':
        c.source=c.source.replace('ACCEPT_ELLIPSOID = False','ACCEPT_ELLIPSOID = True')
        c.source=c.source.replace('ACCEPT_CUBE_ALIGNMENT = False','ACCEPT_CUBE_ALIGNMENT = True')
        c.source=c.source.replace('ACCEPT_GYRO_BIAS = False','ACCEPT_GYRO_BIAS = True')
        c.source=c.source.replace('ACCEPT_GYRO_AXES = False','ACCEPT_GYRO_AXES = True')
        c.source=c.source.replace('ACCEPT_FINAL_CALIBRATION = False','ACCEPT_FINAL_CALIBRATION = True')
try:
    executed = NotebookClient(note,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    errors=[(i,x) for i,c in enumerate(executed.cells) if c.cell_type=='code' for x in c.get('outputs',[]) if x.output_type=='error']
    if errors: raise RuntimeError(repr(errors)[:4000])
    print('All',len(note.cells),'notebook cells executed end-to-end on offline synthetic datasets.')
finally:
    import shutil
    shutil.rmtree(DATA)
    (ROOT/'data'/'calibration.json').unlink(missing_ok=True)
    (ROOT/'data'/'calibration.provenance.json').unlink(missing_ok=True)
