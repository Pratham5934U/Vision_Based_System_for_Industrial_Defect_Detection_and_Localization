"""
03_cnn_classifier.py — EfficientNet-B0 Fine-Tuning + Grad-CAM
================================================================
Trains a fine-tuned EfficientNet-B0 on the NEU Surface Defect dataset.
After training, generates Grad-CAM heatmaps for explainability.

Expected accuracy on NEU: ~93–97%

Produces:
  models/efficientnet_best.pth      — best model weights
  results/training_curves.png       — loss & accuracy curves
  results/confusion_matrix_cnn.png  — confusion matrix
  results/gradcam_<class>.png       — Grad-CAM overlays per class
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/"
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
BATCH_SIZE  = 16       # reduce to 8 if out of memory
EPOCHS      = 30
LR          = 1e-4
WEIGHT_DECAY= 1e-4
NUM_CLASSES = 6        # 6 NEU defect types
IMG_SIZE    = 224

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # must be after ToTensor
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Datasets ──────────────────────────────────────────────────────────────────
print("\n📁 Loading datasets…")
train_dataset = datasets.ImageFolder("data/train", transform=train_tf)
test_dataset  = datasets.ImageFolder("data/test",  transform=val_tf)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

CLASS_NAMES = train_dataset.classes
print(f"   Classes  : {CLASS_NAMES}")
print(f"   Train    : {len(train_dataset)} images")
print(f"   Test     : {len(test_dataset)} images")
print(f"   Batches  : {len(train_loader)} (train)  |  {len(test_loader)} (test)")


# ── Model ─────────────────────────────────────────────────────────────────────
print("\n🧠 Building EfficientNet-B0 model…")
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Freeze all layers except last 2 feature blocks + classifier
for name, param in model.named_parameters():
    if not any(k in name for k in ["features.7", "features.8", "classifier"]):
        param.requires_grad = False

# Replace head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES),
)
model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"   Trainable params : {trainable:,} / {total:,}")


# ── Optimizer + Scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY,
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


# ── Training loop ─────────────────────────────────────────────────────────────
print("\n🚀 Starting training…")
print("-" * 65)

history = {"train_loss": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    correct, total_n = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total_n  += labels.size(0)

    val_acc = correct / total_n
    scheduler.step()

    history["train_loss"].append(avg_loss)
    history["val_acc"].append(val_acc)

    marker = " ✅ (saved)" if val_acc > best_val_acc else ""
    print(f"Epoch {epoch:3d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Val Acc: {val_acc*100:.2f}%{marker}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODELS_DIR / "efficientnet_best.pth")

print("-" * 65)
print(f"\n🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"💾 Best model saved → models/efficientnet_best.pth")


# ── Training curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f0f0f")

epochs_range = range(1, EPOCHS + 1)
ax1.plot(epochs_range, history["train_loss"], color="#667eea", linewidth=2, label="Train Loss")
ax1.set_title("Training Loss", color="white", fontweight="bold")
ax1.set_xlabel("Epoch", color="white"); ax1.set_ylabel("Loss", color="white")
ax1.tick_params(colors="white"); ax1.spines[:].set_visible(False)
ax1.set_facecolor("#1a1a2e"); ax1.legend(facecolor="#1a1a2e", labelcolor="white")

ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], color="#f093fb", linewidth=2, label="Val Accuracy")
ax2.axhline(y=best_val_acc * 100, color="#43e97b", linestyle="--", linewidth=1, label=f"Best: {best_val_acc*100:.2f}%")
ax2.set_title("Validation Accuracy", color="white", fontweight="bold")
ax2.set_xlabel("Epoch", color="white"); ax2.set_ylabel("Accuracy (%)", color="white")
ax2.tick_params(colors="white"); ax2.spines[:].set_visible(False)
ax2.set_facecolor("#1a1a2e"); ax2.legend(facecolor="#1a1a2e", labelcolor="white")

fig.patch.set_facecolor("#0f0f0f")
plt.suptitle("EfficientNet-B0 Training — NEU Surface Defect",
             color="white", fontsize=14, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"\n📈 Training curves saved → {out}")
plt.close()


# ── Final evaluation & confusion matrix ───────────────────────────────────────
print("\n📊 Final evaluation on test set…")
model.load_state_dict(torch.load(MODELS_DIR / "efficientnet_best.pth", map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f0f0f")
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=[c.replace("_", "\n") for c in CLASS_NAMES],
    yticklabels=[c.replace("_", "\n") for c in CLASS_NAMES],
    cmap="Purples", ax=ax, linewidths=0.5, linecolor="#333",
)
ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
ax.set_title(f"Confusion Matrix — EfficientNet-B0\nBest Acc: {best_val_acc*100:.2f}%",
             color="white", fontweight="bold")
ax.tick_params(colors="white"); fig.patch.set_facecolor("#0f0f0f")
plt.tight_layout()
out = RESULTS_DIR / "confusion_matrix_cnn.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"🖼  Confusion matrix saved → {out}")
plt.close()


# ── Grad-CAM Visualisation ────────────────────────────────────────────────────
print("\n🔬 Generating Grad-CAM heatmaps…")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    for cls_idx, cls in enumerate(CLASS_NAMES):
        cls_path = Path("data/test") / cls
        if not cls_path.exists():
            continue
        imgs_in_cls = sorted([p for p in cls_path.glob("*") if p.suffix.lower() in {".jpg", ".bmp", ".png"}])
        if not imgs_in_cls:
            continue

        img_path = imgs_in_cls[0]
        from PIL import Image as PILImage
        pil_img  = PILImage.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_np   = np.array(pil_img) / 255.0

        input_tensor = val_tf(PILImage.open(img_path)).unsqueeze(0).to(DEVICE)
        targets = [ClassifierOutputTarget(cls_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0f0f0f")
        axes[0].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
        axes[0].set_title("Original", color="white")
        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", color="white")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", color="white")
        for ax in axes:
            ax.axis("off")
        fig.patch.set_facecolor("#0f0f0f")
        plt.suptitle(f"Grad-CAM — {cls.replace('_', ' ').title()}",
                     color="white", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = RESULTS_DIR / f"gradcam_{cls}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        print(f"   {cls:22s} → {out}")
        plt.close()

    print("✅ Grad-CAM heatmaps saved!")

except ImportError:
    print("⚠️  pytorch-grad-cam not installed. Run: pip install grad-cam")

print("\n✅ CNN training and evaluation complete!")
