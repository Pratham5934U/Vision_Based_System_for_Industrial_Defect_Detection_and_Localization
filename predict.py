"""
predict.py
----------
Single-image inference using the trained EfficientNet-B0 model.

Usage (command line):
    python predict.py path/to/image.bmp

Usage (as module):
    from predict import load_model, predict
    model = load_model()
    result = predict("path/to/image.bmp", model)
    print(result)
"""

import sys
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


# ── Config ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CHECKPOINT  = "models/efficientnet_best.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224

# Confidence threshold: below this → UNCERTAIN (model not sure)
CONFIDENCE_THRESHOLD = 0.70

# ImageNet normalization (EfficientNet expects this)
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # NEU is grayscale; model needs 3ch
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])


# ── Model ────────────────────────────────────────────────────────────────────
def load_model(checkpoint: str = CHECKPOINT) -> torch.nn.Module:
    """Load the trained EfficientNet-B0 model from a checkpoint file."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            f"Run notebook 03_cnn_classifier.py first to train and save the model."
        )

    model = models.efficientnet_b0(weights=None)
    # Match the classifier head used during training
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, len(CLASS_NAMES)),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image_path: str, model: torch.nn.Module) -> dict:
    """
    Run inference on a single image.

    Returns a dict with:
        - verdict      : "FAIL" | "UNCERTAIN"
        - defect_type  : predicted class name
        - confidence   : float [0, 1]
        - all_probs    : dict of {class_name: probability}
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path)
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # shape: [1, 3, H, W]

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]      # shape: [num_classes]

    confidence      = probs.max().item()
    predicted_idx   = probs.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]

    # Simple pass/fail logic
    verdict = "FAIL" if confidence >= CONFIDENCE_THRESHOLD else "UNCERTAIN"

    return {
        "verdict"    : verdict,
        "defect_type": predicted_class,
        "confidence" : round(confidence, 4),
        "all_probs"  : {c: round(p.item(), 4) for c, p in zip(CLASS_NAMES, probs)},
    }


def generate_gradcam(image_path: str, model: torch.nn.Module, class_name: str = None) -> tuple:
    """
    Generate Grad-CAM attention heatmap and defect overlay for an image.
    Returns:
        (heatmap_rgb, overlay_rgb) as uint8 numpy arrays.
    """
    import cv2
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    pil_img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(pil_img) / 255.0

    tensor = TRANSFORM(Image.open(img_path)).unsqueeze(0).to(DEVICE)

    if class_name is not None and class_name in CLASS_NAMES:
        cls_idx = CLASS_NAMES.index(class_name)
    else:
        with torch.no_grad():
            logits = model(tensor)
            cls_idx = logits.argmax().item()

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(cls_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    # Colored heatmap (Jet)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Defect overlay on original image
    overlay_rgb = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    return heatmap_rgb, overlay_rgb


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Loading model from: {CHECKPOINT}")
    model = load_model()

    print(f"Running inference on: {image_path}\n")
    result = predict(image_path, model)

    print(f"  Verdict     : {result['verdict']}")
    print(f"  Defect Type : {result['defect_type']}")
    print(f"  Confidence  : {result['confidence']*100:.1f}%")
    print(f"\n  All Probabilities:")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:20s}  {prob*100:5.1f}%  {bar}")
