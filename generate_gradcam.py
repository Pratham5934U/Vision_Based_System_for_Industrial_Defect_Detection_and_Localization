"""
generate_gradcam.py
-------------------
Generates and saves Grad-CAM heatmaps for sample test images of each defect class
using the trained EfficientNet-B0 model saved in models/efficientnet_best.pth.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import models, transforms

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CHECKPOINT  = Path("models/efficientnet_best.pth")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading trained model from:", CHECKPOINT)
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, len(CLASS_NAMES)),
)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
print("\nGenerating Grad-CAM heatmaps...")
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    saved_count = 0
    for cls_idx, cls in enumerate(CLASS_NAMES):
        cls_path = Path("data/test") / cls
        if not cls_path.exists():
            print(f"   [!] Folder not found: {cls_path}")
            continue
        imgs_in_cls = sorted([p for p in cls_path.glob("*") if p.suffix.lower() in {".jpg", ".bmp", ".png"}])
        if not imgs_in_cls:
            print(f"   [!] No images in: {cls_path}")
            continue

        img_path = imgs_in_cls[0]
        pil_img  = PILImage.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np   = np.array(pil_img) / 255.0

        input_tensor = val_tf(PILImage.open(img_path)).unsqueeze(0).to(DEVICE)
        targets = [ClassifierOutputTarget(cls_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0f0f0f")
        axes[0].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
        axes[0].set_title("Original", color="white", fontsize=12)
        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=12)
        axes[2].imshow(overlay)
        axes[2].set_title("Defect Overlay", color="white", fontsize=12)
        for ax in axes:
            ax.axis("off")
        fig.patch.set_facecolor("#0f0f0f")
        plt.suptitle(f"Grad-CAM Explanation -- {cls.replace('_', ' ').title()}",
                     color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = RESULTS_DIR / f"gradcam_{cls}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        print(f"   [OK] Saved: {cls:20s} -> {out}")
        plt.close()
        saved_count += 1

    print(f"\nSuccessfully saved {saved_count} Grad-CAM heatmaps in {RESULTS_DIR}/")

except Exception as e:
    print(f"[ERROR] Error generating Grad-CAM: {e}")
