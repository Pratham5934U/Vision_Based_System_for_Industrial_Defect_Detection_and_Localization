import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')
import torch
from pathlib import Path
from anomalib.models import Patchcore
from anomaly_predict import predict_anomaly, get_model_threshold

ckpt = Path('results/mvtec_patchcore/metal_nut/Patchcore/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt')
model = Patchcore.load_from_checkpoint(str(ckpt))
model.eval()

print(f'Model threshold (learned): {get_model_threshold(model):.4f}')
print(f'Using default cutoff     : 14.8')
print()

tests = [
    ('GOOD',    'mv dataset/metal_nut/test/good/000.png',    'NORMAL'),
    ('GOOD',    'mv dataset/metal_nut/test/good/005.png',    'NORMAL'),
    ('GOOD',    'mv dataset/metal_nut/test/good/010.png',    'NORMAL'),
    ('GOOD',    'mv dataset/metal_nut/test/good/015.png',    'NORMAL'),
    ('SCRATCH', 'mv dataset/metal_nut/test/scratch/000.png', 'ANOMALY'),
    ('SCRATCH', 'mv dataset/metal_nut/test/scratch/002.png', 'ANOMALY'),
    ('SCRATCH', 'mv dataset/metal_nut/test/scratch/010.png', 'ANOMALY'),
    ('BENT',    'mv dataset/metal_nut/test/bent/000.png',    'ANOMALY'),
    ('BENT',    'mv dataset/metal_nut/test/bent/010.png',    'ANOMALY'),
    ('COLOR',   'mv dataset/metal_nut/test/color/000.png',   'ANOMALY'),
    ('COLOR',   'mv dataset/metal_nut/test/color/002.png',   'ANOMALY'),
]

correct = 0
total = 0
for label, path, expected in tests:
    if not Path(path).exists():
        continue
    res = predict_anomaly(path, model)
    ok = res['verdict'] == expected
    if ok:
        correct += 1
    total += 1
    status = 'OK   ' if ok else 'WRONG'
    print(f'[{status}] {label:7s} | raw={res["raw_score"]:6.2f} | score={res["anomaly_score"]:.3f} | {res["verdict"]:7s} | conf={res["confidence_pct"]:.1f}%  (expected {expected})')

print()
print(f'Accuracy: {correct}/{total}')
