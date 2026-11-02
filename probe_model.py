import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

ckpt = Path('results/mvtec_patchcore/metal_nut/Patchcore/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt')
from anomalib.models import Patchcore
model = Patchcore.load_from_checkpoint(str(ckpt))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def score_images(folder):
    scores = []
    folder = Path(folder)
    if not folder.exists():
        return scores
    for img_path in sorted(folder.glob('*.png')):
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model.model(tensor)
        scores.append(float(out.pred_score.squeeze()))
    return scores

print("Scoring ALL good training images...")
good_train = score_images('mv dataset/metal_nut/train/good')
print(f"Train good: n={len(good_train)}, min={min(good_train):.3f}, max={max(good_train):.3f}, mean={np.mean(good_train):.3f}, std={np.std(good_train):.3f}")
print(f"  95th percentile: {np.percentile(good_train, 95):.3f}")
print(f"  99th percentile: {np.percentile(good_train, 99):.3f}")
print(f"  max+small margin: {max(good_train)*1.05:.3f}")
print()

print("Scoring ALL good TEST images...")
good_test = score_images('mv dataset/metal_nut/test/good')
print(f"Test good:  n={len(good_test)}, min={min(good_test):.3f}, max={max(good_test):.3f}, mean={np.mean(good_test):.3f}")
print()

print("Scoring defect test images...")
for cat in ['scratch', 'bent', 'color']:
    scores = score_images(f'mv dataset/metal_nut/test/{cat}')
    if scores:
        print(f"Test {cat}: n={len(scores)}, min={min(scores):.3f}, max={max(scores):.3f}, mean={np.mean(scores):.3f}")

print()
# Proposed threshold: 95th or 99th percentile of training good images
thresh_95 = np.percentile(good_train, 95)
thresh_99 = np.percentile(good_train, 99)
thresh_max = max(good_train) * 1.05
print(f"Candidate thresholds:")
print(f"  Train-good 95th pct: {thresh_95:.4f}")
print(f"  Train-good 99th pct: {thresh_99:.4f}")
print(f"  Train-good max*1.05: {thresh_max:.4f}")
print(f"  Model post-processor: 11.6959")
