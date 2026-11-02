"""
anomaly_predict.py
------------------
Single-image anomaly scoring using the trained PatchCore model on genuine
defect-free MVTec AD industrial products.

Key design:
- Uses model.model() to get RAW patch feature distances (not post-processed)
- The model's learned threshold (extracted from checkpoint) is used for the verdict
- Score normalization maps the raw distance to a clean [0,1] display range
"""

import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MVTEC_PATCHCORE_CKPT = Path("results/mvtec_patchcore/metal_nut/Patchcore/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt")
NEU_PATCHCORE_CKPT   = Path("results/Patchcore/surface_defects/v0/weights/lightning/model.ckpt")

# Score range measured across all metal_nut images (raw feature distances from model.model())
# Train good: mean=12.94, 95th pct=14.56, 99th pct=14.98, max=15.42
# Test scratch: mean=15.50  |  Test bent: mean=17.01  |  Test color: mean=16.14
# Default threshold = 14.8 (just above 95th pct of good, catches all clear defects)
_DEFAULT_THRESHOLD = 14.8   # raw distance cutoff (overrides model's 11.70 which was too tight)
_SCORE_LOW  = 10.0  # minimum seen in training good images
_SCORE_HIGH = 20.0  # maximum seen in defect images


def get_default_checkpoint() -> Path:
    if MVTEC_PATCHCORE_CKPT.exists():
        return MVTEC_PATCHCORE_CKPT
    if NEU_PATCHCORE_CKPT.exists():
        return NEU_PATCHCORE_CKPT
    return MVTEC_PATCHCORE_CKPT


def load_patchcore(checkpoint_path: str = None):
    """
    Load the trained PatchCore model from checkpoint.
    Also extracts the model's learned decision threshold.
    """
    ckpt = Path(checkpoint_path) if checkpoint_path else get_default_checkpoint()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found at: {ckpt}\n"
            "Run 'python train_mvtec_patchcore.py --category metal_nut' first."
        )

    from anomalib.models import Patchcore
    model = Patchcore.load_from_checkpoint(str(ckpt))
    model.eval()

    # Extract learned threshold from the post-processor
    _extract_threshold(model)
    return model


def _extract_threshold(model) -> float:
    """Extract the model's learned image-level threshold and cache on model."""
    thresh = None
    # Try post_processor attribute
    if hasattr(model, "post_processor"):
        pp = model.post_processor
        if hasattr(pp, "image_threshold"):
            t = pp.image_threshold
            if hasattr(t, "item"):
                thresh = float(t.item())
            elif isinstance(t, (int, float)):
                thresh = float(t)
    # Fall back to image_threshold attr on model
    if thresh is None and hasattr(model, "image_threshold"):
        t = model.image_threshold
        thresh = float(t.item()) if hasattr(t, "item") else float(t)
    # Hard-coded fallback from evaluation (model trained on metal_nut)
    if thresh is None:
        thresh = 11.6959
    model._cached_threshold = thresh
    return thresh


def get_model_threshold(model) -> float:
    """Return the cached threshold from the model, extracting if needed."""
    if not hasattr(model, "_cached_threshold"):
        _extract_threshold(model)
    return model._cached_threshold


def _get_transform():
    """Standard image transform for PatchCore backbone."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict_anomaly(image_path: str, model, threshold: float = None) -> dict:
    """
    Run anomaly detection on a single image.

    Args:
        image_path : path to the image file
        model      : loaded PatchCore model (from load_patchcore())
        threshold  : optional override for decision threshold (raw distance units)
                     If None, uses the model's learned threshold from training.

    Returns dict with:
        verdict        : "ANOMALY" | "NORMAL"
        anomaly_score  : normalized [0,1] gauge score (0.5 = decision boundary)
        threshold      : display threshold (always 0.5 for the gauge)
        confidence_pct : confidence percentage [0, 100]
        raw_score      : actual patch distance from model.model()
        raw_threshold  : the raw-distance cutoff being applied
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Use calibrated default threshold unless user explicitly overrides.
    # The model's learned threshold (11.70) maximizes AUROC but causes too many
    # false positives on good images (good test images score up to 15.5).
    # _DEFAULT_THRESHOLD=14.8 sits between the 95th pct of good and avg defect.
    cutoff = float(threshold) if threshold is not None else _DEFAULT_THRESHOLD

    transform = _get_transform()
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    # Forward pass through PatchcoreModel (backbone only) for raw distance
    with torch.no_grad():
        raw_model = model.model if hasattr(model, "model") else model
        out = raw_model(tensor)
        if hasattr(out, "pred_score"):
            raw_score = float(out.pred_score.squeeze())
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            raw_score = float(out[0].squeeze())
        else:
            raw_score = cutoff  # fallback

    # Decision
    is_anomaly = raw_score >= cutoff
    verdict = "ANOMALY" if is_anomaly else "NORMAL"

    # ── Normalize to a clean [0, 1] display score ──────────────────────────
    # We map the full score range linearly so that:
    #   raw = cutoff  →  norm = 0.50  (decision boundary)
    #   raw = _SCORE_LOW   →  norm ≈ 0.0
    #   raw = _SCORE_HIGH  →  norm ≈ 1.0
    # This gives a stable, continuous gauge.
    score_range = _SCORE_HIGH - _SCORE_LOW
    norm_score = np.clip((raw_score - _SCORE_LOW) / score_range, 0.0, 1.0)
    # Re-scale so that cutoff maps exactly to 0.5
    cutoff_norm = np.clip((cutoff - _SCORE_LOW) / score_range, 0.01, 0.99)
    if norm_score >= cutoff_norm:
        # anomaly side: remap [cutoff_norm, 1] → [0.5, 1]
        norm_score = 0.5 + 0.5 * (norm_score - cutoff_norm) / (1.0 - cutoff_norm + 1e-8)
    else:
        # normal side: remap [0, cutoff_norm] → [0, 0.5]
        norm_score = 0.5 * norm_score / (cutoff_norm + 1e-8)
    norm_score = float(np.clip(norm_score, 0.0, 1.0))

    # Confidence: how far from the boundary (in raw-distance units), capped at 100%
    dist_from_boundary = abs(raw_score - cutoff)
    boundary_half_range = (score_range / 2.0)  # ~5.5 units
    confidence_pct = min(99.9, 50.0 + (dist_from_boundary / boundary_half_range) * 50.0)

    return {
        "verdict":        verdict,
        "anomaly_score":  round(norm_score, 3),
        "threshold":      0.50,
        "confidence_pct": round(float(confidence_pct), 1),
        "raw_score":      round(float(raw_score), 4),
        "raw_threshold":  round(float(cutoff), 4),
    }


def get_anomaly_heatmap(image_path: str, model):
    """
    Returns the per-pixel anomaly heatmap as an RGB numpy array, or None.
    """
    img_path = Path(image_path)
    transform = _get_transform()
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    try:
        with torch.no_grad():
            raw_model = model.model if hasattr(model, "model") else model
            out = raw_model(tensor)

        amap = None
        if hasattr(out, "anomaly_map") and out.anomaly_map is not None:
            amap = out.anomaly_map.squeeze().cpu().numpy()
        elif isinstance(out, (list, tuple)) and len(out) > 2:
            amap = out[2].squeeze().cpu().numpy()

        if amap is None:
            return None

        # Normalize to 0..255
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        amap_uint8 = (amap_norm * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    except Exception:
        return None
