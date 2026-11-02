"""
02_classical_cv.py — Classical Computer Vision Baseline
=========================================================
Extracts hand-crafted features (Canny edges + GLCM texture) and trains
a Random Forest classifier. This is the baseline to beat with deep learning.

Expected accuracy on NEU: ~72–78%

Produces:
  models/classical_cv_model.pkl   — trained Random Forest + scaler
  results/classical_cv_report.txt — classification report
  results/confusion_matrix_cv.png — confusion matrix heatmap
  results/feature_importance.png  — top feature importances
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("data/train")
TEST_DIR    = Path("data/test")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
IMG_SIZE    = (200, 200)
N_TREES     = 200

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(image_path: Path) -> np.ndarray:
    """
    Extract hand-crafted features from one grayscale image.

    Features:
        [0]      edge_density     — fraction of Canny edge pixels
        [1]      std_dev          — standard deviation of pixel values (texture roughness)
        [2]      glcm_contrast    — GLCM contrast (local intensity variation)
        [3]      glcm_homogeneity — GLCM homogeneity (closeness of distribution to diagonal)
        [4]      glcm_energy      — GLCM energy (uniformity)
        [5]      glcm_correlation — GLCM correlation (linear dependency)
        [6:38]   histogram        — 32-bin normalized intensity histogram
    """
    from skimage.feature import graycomatrix, graycoprops

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(37, dtype=np.float32)

    img = cv2.resize(img, IMG_SIZE)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Feature 1: Edge density
    edges = cv2.Canny(enhanced, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Feature 2: Std deviation (roughness)
    std_dev = float(np.std(enhanced))

    # Feature 3–6: GLCM texture descriptors
    glcm = graycomatrix(
        enhanced // 4,          # quantize to 64 levels
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    contrast    = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy      = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())

    # Feature 7: Normalized histogram (32 bins)
    hist = cv2.calcHist([enhanced], [0], None, [32], [0, 256]).flatten()
    hist /= hist.sum() + 1e-8

    return np.array(
        [edge_density, std_dev, contrast, homogeneity, energy, correlation]
        + hist.tolist(),
        dtype=np.float32,
    )


FEATURE_NAMES = (
    ["edge_density", "std_dev", "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
    + [f"hist_bin_{i}" for i in range(32)]
)


def build_feature_matrix(data_dir: Path, classes: list[str], class_to_idx: dict):
    """Load all images from a directory and extract features."""
    X, y = [], []
    total = sum(len(list((data_dir / cls).glob("*"))) for cls in classes)
    done  = 0
    for cls in classes:
        img_paths = sorted((data_dir / cls).glob("*"))
        img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}]
        for img_path in img_paths:
            X.append(extract_features(img_path))
            y.append(class_to_idx[cls])
            done += 1
            if done % 100 == 0:
                print(f"   Processed {done}/{total} images…", end="\r")
    print()
    return np.array(X, dtype=np.float32), np.array(y)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CLASSICAL CV BASELINE — GLCM + Edges + Random Forest")
    print("=" * 60)

    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(f"\nClasses ({len(classes)}): {classes}\n")

    # ── Feature extraction ────────────────────────────────────────────────
    print("⚙️  Extracting training features…")
    X_train, y_train = build_feature_matrix(TRAIN_DIR, classes, class_to_idx)
    print(f"   Train feature matrix: {X_train.shape}  (samples × features)")

    print("⚙️  Extracting test features…")
    X_test,  y_test  = build_feature_matrix(TEST_DIR,  classes, class_to_idx)
    print(f"   Test  feature matrix: {X_test.shape}")

    # ── Normalisation ─────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n🌲 Training Random Forest ({N_TREES} trees)…")
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)
    print("   Training done ✅")

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes)

    print(f"\n📊 Test Accuracy: {acc*100:.2f}%\n")
    print(report)

    report_path = RESULTS_DIR / "classical_cv_report.txt"
    report_path.write_text(f"Test Accuracy: {acc*100:.2f}%\n\n{report}")
    print(f"📄 Report saved → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f0f0f")
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=[c.replace("_", "\n") for c in classes],
        yticklabels=[c.replace("_", "\n") for c in classes],
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="#333",
    )
    ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("Actual", color="white")
    ax.set_title(f"Confusion Matrix — Classical CV (RF)\nAccuracy: {acc*100:.2f}%",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white"); fig.patch.set_facecolor("#0f0f0f")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix_cv.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"🖼  Confusion matrix saved → {cm_path}")
    plt.close()

    # ── Feature importances ───────────────────────────────────────────────
    importances = clf.feature_importances_
    top_n = 15
    top_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f0f")
    colors = ["#667eea" if i < 6 else "#f093fb" for i in top_idx]
    ax.barh(
        [FEATURE_NAMES[i] for i in top_idx][::-1],
        importances[top_idx][::-1],
        color=colors[::-1], edgecolor="none",
    )
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 color="white", fontweight="bold")
    ax.set_xlabel("Importance", color="white")
    ax.tick_params(colors="white"); ax.spines[:].set_visible(False)
    ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#0f0f0f")
    plt.tight_layout()
    fi_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"🖼  Feature importances saved → {fi_path}")
    plt.close()

    # ── Save model ────────────────────────────────────────────────────────
    model_bundle = {"classifier": clf, "scaler": scaler, "classes": classes}
    model_path   = MODELS_DIR / "classical_cv_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n💾 Model saved → {model_path}")
    print("\n✅ Classical CV baseline complete!")
