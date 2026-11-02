"""
01_eda.py — Exploratory Data Analysis
======================================
Run this FIRST to understand your dataset before building models.

Produces:
  results/sample_images.png    — grid of one sample per class
  results/class_distribution.png — bar chart of images per class
  results/pixel_stats.png      — mean brightness per class
  results/eda_summary.txt      — text summary
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/train")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Discover classes ───────────────────────────────────────────────────────
print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS — NEU Surface Defect Dataset")
print("=" * 60)

classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"\n✅ Found {len(classes)} classes:")
for cls in classes:
    print(f"   • {cls}")

# ── 2. Count images per class ──────────────────────────────────────────────
counts = {}
for cls in classes:
    imgs = list((DATA_DIR / cls).glob("*"))
    imgs = [i for i in imgs if i.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}]
    counts[cls] = len(imgs)
    print(f"   {cls:22s}: {len(imgs):5d} images")

total = sum(counts.values())
print(f"\n   Total training images: {total}")

# ── 3. Check image shapes ──────────────────────────────────────────────────
sample_img_path = list((DATA_DIR / classes[0]).glob("*"))[0]
sample_img = cv2.imread(str(sample_img_path))
if sample_img is not None:
    print(f"\n📐 Image shape (H, W, C): {sample_img.shape}")
    print(f"   dtype: {sample_img.dtype}")
    print(f"   pixel range: [{sample_img.min()}, {sample_img.max()}]")

# ── 4. Plot: sample images grid ────────────────────────────────────────────
print("\n📸 Plotting sample images grid…")
fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="#0f0f0f")
fig.suptitle("NEU Surface Defect — Sample Images per Class",
             fontsize=16, color="white", fontweight="bold", y=0.98)

for ax, cls in zip(axes.flatten(), classes):
    imgs = sorted((DATA_DIR / cls).glob("*"))
    imgs = [i for i in imgs if i.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}]
    if not imgs:
        ax.axis("off")
        continue
    img = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap="gray")
    ax.set_title(cls.replace("_", " ").title(), color="white", fontsize=12, fontweight="bold")
    ax.axis("off")
    # Draw border
    for spine in ax.spines.values():
        spine.set_edgecolor("#667eea")
        spine.set_linewidth(2)

plt.tight_layout()
out = RESULTS_DIR / "sample_images.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"   Saved → {out}")

plt.close()

# ── 5. Plot: class distribution bar chart ──────────────────────────────────
print("\n📊 Plotting class distribution…")
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f0f")
colors = ["#667eea", "#f093fb", "#4facfe", "#43e97b", "#f7971e", "#fa709a"]
bars = ax.bar(
    [c.replace("_", "\n") for c in classes],
    counts.values(),
    color=colors, edgecolor="none", alpha=0.9
)

for bar, count in zip(bars, counts.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            str(count), ha="center", va="bottom", color="white", fontsize=11)

ax.set_title("Class Distribution (Training Set)", color="white", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Images", color="white")
ax.tick_params(colors="white")
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#0f0f0f")
ax.spines[:].set_visible(False)
ax.set_ylim(0, max(counts.values()) * 1.15)

plt.tight_layout()
out = RESULTS_DIR / "class_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"   Saved → {out}")

plt.close()

# ── 6. Pixel statistics per class ─────────────────────────────────────────
print("\n📈 Computing pixel statistics per class…")
stats = defaultdict(lambda: {"means": [], "stds": []})

for cls in classes:
    imgs = sorted((DATA_DIR / cls).glob("*"))
    imgs = [i for i in imgs if i.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}]
    for img_path in imgs[:50]:   # sample 50 per class for speed
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stats[cls]["means"].append(np.mean(img))
            stats[cls]["stds"].append(np.std(img))

class_means = [np.mean(stats[cls]["means"]) for cls in classes]
class_stds  = [np.mean(stats[cls]["stds"])  for cls in classes]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0f0f0f")

ax1.bar([c.replace("_", "\n") for c in classes], class_means, color=colors, alpha=0.9)
ax1.set_title("Mean Pixel Brightness per Class", color="white", fontweight="bold")
ax1.set_ylabel("Mean Brightness (0–255)", color="white")
ax1.tick_params(colors="white"); ax1.spines[:].set_visible(False)
ax1.set_facecolor("#1a1a2e")

ax2.bar([c.replace("_", "\n") for c in classes], class_stds, color=colors, alpha=0.9)
ax2.set_title("Mean Pixel Std Dev per Class", color="white", fontweight="bold")
ax2.set_ylabel("Std Dev (texture roughness)", color="white")
ax2.tick_params(colors="white"); ax2.spines[:].set_visible(False)
ax2.set_facecolor("#1a1a2e")

fig.patch.set_facecolor("#0f0f0f")
plt.tight_layout()
out = RESULTS_DIR / "pixel_stats.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"   Saved → {out}")

plt.close()

# ── 7. CLAHE demo: before vs after ────────────────────────────────────────
print("\n🔍 Demonstrating CLAHE contrast enhancement…")
sample_path = list((DATA_DIR / classes[0]).glob("*"))[0]
raw = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
enhanced = clahe.apply(raw)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0f0f0f")
axes[0].imshow(raw,      cmap="gray"); axes[0].set_title("Original",        color="white")
axes[1].imshow(enhanced, cmap="gray"); axes[1].set_title("After CLAHE",     color="white")
for ax in axes:
    ax.axis("off")
    ax.set_facecolor("#0f0f0f")
fig.patch.set_facecolor("#0f0f0f")
plt.suptitle("CLAHE Contrast Enhancement", color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
out = RESULTS_DIR / "clahe_demo.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print(f"   Saved → {out}")

plt.close()

# ── 8. Summary ────────────────────────────────────────────────────────────
summary_lines = [
    "EDA SUMMARY — NEU Surface Defect Dataset",
    "=" * 50,
    f"Total training images : {total}",
    f"Number of classes     : {len(classes)}",
    f"Classes               : {', '.join(classes)}",
    f"Image shape           : {sample_img.shape if sample_img is not None else 'N/A'}",
    "",
    "Per-class stats:",
]
for cls in classes:
    m = np.mean(stats[cls]["means"])
    s = np.mean(stats[cls]["stds"])
    summary_lines.append(f"  {cls:22s}  count: {counts[cls]:3d}  mean_brightness: {m:.1f}  std: {s:.1f}")

summary = "\n".join(summary_lines)
print("\n" + summary)

summary_path = RESULTS_DIR / "eda_summary.txt"
summary_path.write_text(summary)
print(f"\n📄 Summary saved → {summary_path}")
print("\n✅ EDA complete! Check the results/ folder for all plots.")
