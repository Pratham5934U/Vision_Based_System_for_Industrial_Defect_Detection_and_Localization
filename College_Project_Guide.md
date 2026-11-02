# 🏭 AI Quality Inspection — College Project Guide
### Just your laptop + Python + photos. Nothing else needed.

---

## What You're Actually Building

A Python project that:
1. **Takes images** from a folder (your dataset)
2. **Detects defects** using 2–3 approaches (classical CV → CNN → anomaly detection)
3. **Shows results** in a simple Streamlit web UI
4. **Logs results** to a CSV file

That's it. No cameras, no conveyor belts, no servers.

---

## Step 1: Get Your Dataset

Download **one** of these — free, no sign-up needed for most:

| Dataset | What it is | Size | Download |
|---|---|---|---|
| **NEU Surface Defect** ✅ Recommended | Steel surface defects, 6 classes | ~60MB | [Kaggle link](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| Casting Product Defect | Metal castings, good vs defective | ~200MB | [Kaggle link](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) |
| MVTec AD | Multi-category anomaly detection | ~4GB | [Official site](https://www.mvtec.com/company/research/datasets/mvtec-ad) |

**NEU is the best pick** — small size, 6 clear defect types, grayscale, well-studied in papers.

```
After download, your folder should look like:
data/
├── train/
│   ├── crazing/        (300 images)
│   ├── inclusion/      (300 images)
│   ├── patches/        (300 images)
│   ├── pitted_surface/ (300 images)
│   ├── rolled-in_scale/(300 images)
│   └── scratches/      (300 images)
└── test/
    └── (same structure, ~20% of images)
```

Use this script to split automatically:
```python
# split_dataset.py
import os, shutil, random
from pathlib import Path

SRC = "data/NEU-DET/images"   # adjust to your downloaded folder
TRAIN_RATIO = 0.8

for class_folder in Path(SRC).iterdir():
    images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.bmp"))
    random.shuffle(images)
    split = int(len(images) * TRAIN_RATIO)
    
    for i, img in enumerate(images):
        dest_split = "train" if i < split else "test"
        dest = Path(f"data/{dest_split}/{class_folder.name}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest / img.name)

print("Done! Check your data/ folder.")
```

---

## Step 2: Project Folder Structure

Keep it simple:

```
quality-inspection/
│
├── data/                   ← your images go here
│   ├── train/
│   └── test/
│
├── models/                 ← saved model weights (.pth files)
│
├── results/                ← output images, CSV logs
│
├── 01_eda.ipynb            ← explore your dataset
├── 02_classical_cv.ipynb   ← classical baseline
├── 03_cnn_classifier.ipynb ← deep learning model
├── 04_anomaly.ipynb        ← anomaly detection
│
├── app.py                  ← Streamlit dashboard
├── predict.py              ← run prediction on a single image
└── requirements.txt
```

---

## Step 3: Install Everything

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-learn scikit-image
pip install matplotlib seaborn
pip install streamlit
pip install albumentations
pip install anomalib
pip install grad-cam
pip install pandas pillow tqdm
```

> 💡 **No GPU?** That's fine. EfficientNet on CPU takes ~2–3 min/epoch for NEU. Train for 20–30 epochs overnight if needed. Or use Google Colab (free GPU) to train and download the `.pth` file.

---

## Step 4: Explore Your Data (EDA Notebook)

```python
# 01_eda.ipynb
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/train")
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

# 1. Count images per class
counts = {c: len(list((DATA_DIR/c).glob("*"))) for c in classes}
print("Images per class:", counts)

# 2. Show sample images from each class
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, cls in zip(axes.flatten(), classes):
    img_path = list((DATA_DIR/cls).glob("*"))[0]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')
    ax.set_title(cls)
    ax.axis('off')
plt.tight_layout()
plt.savefig("results/sample_images.png")
plt.show()

# 3. Check image dimensions
sample = cv2.imread(str(list((DATA_DIR/classes[0]).glob("*"))[0]))
print(f"Image size: {sample.shape}")   # e.g. (200, 200, 3) for NEU
```

---

## Step 5: Classical CV Baseline

Build this first — it's fast and shows you understand the basics.

```python
# 02_classical_cv.ipynb
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

def extract_features(image_path):
    """Extract hand-crafted features from one image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    
    # CLAHE — improves contrast, reveals subtle defects
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    # Feature 1: Edge density (how many edges in the image)
    edges = cv2.Canny(enhanced, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Feature 2: Standard deviation of pixel values (texture roughness)
    std_dev = np.std(enhanced)
    
    # Feature 3: GLCM texture features
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(enhanced // 4, distances=[1], angles=[0], 
                        levels=64, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    
    # Feature 4: Histogram spread (defects change intensity distribution)
    hist = cv2.calcHist([enhanced], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()   # normalize
    
    return np.array([edge_density, std_dev, contrast, homogeneity, energy] + 
                    hist.tolist())

# Build feature matrix
DATA_DIR = Path("data/train")
classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
class_to_idx = {c: i for i, c in enumerate(classes)}

X_train, y_train = [], []
for cls in classes:
    for img_path in (DATA_DIR/cls).glob("*"):
        X_train.append(extract_features(img_path))
        y_train.append(class_to_idx[cls])

X_train = np.array(X_train)

# Train a Random Forest classifier on these features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_scaled, y_train)

# Evaluate on test set
X_test, y_test = [], []
for cls in classes:
    for img_path in Path("data/test", cls).glob("*"):
        X_test.append(extract_features(img_path))
        y_test.append(class_to_idx[cls])

X_test_scaled = scaler.transform(X_test)
y_pred = clf.predict(X_test_scaled)

print(classification_report(y_test, y_pred, target_names=classes))
# Expected: ~70-78% accuracy — good baseline!
```

---

## Step 6: CNN Classifier (EfficientNet-B0)

This is your main model. Fine-tune a pretrained network — no training from scratch needed.

```python
# 03_cnn_classifier.ipynb
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

# --- Config ---
DATA_DIR   = "data/"
BATCH_SIZE = 16          # lower to 8 if you get memory errors
EPOCHS     = 30
LR         = 1e-4
NUM_CLASSES = 6          # 6 defect types in NEU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {DEVICE}")

# --- Data ---
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # EfficientNet expects 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("data/train", transform=train_tf)
test_dataset  = datasets.ImageFolder("data/test",  transform=val_tf)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = train_dataset.classes
print("Classes:", CLASS_NAMES)

# --- Model ---
model = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Freeze all layers except the last block + classifier
for name, param in model.named_parameters():
    if "features.7" not in name and "features.8" not in name and "classifier" not in name:
        param.requires_grad = False

# Replace the head
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

# --- Training ---
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    
    val_acc = correct / total
    scheduler.step()
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/efficientnet_best.pth")
        print(f"  ✅ Saved best model (acc={val_acc:.4f})")

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
# Expected on NEU: ~93-97%
```

---

## Step 7: Grad-CAM — Visualize What the Model Sees

This is important for your report and demo. Shows which part of the image the model focused on.

```python
# (add to 03_cnn_classifier.ipynb)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt

# Load best model
model.load_state_dict(torch.load("models/efficientnet_best.pth", map_location=DEVICE))
model.eval()

# Target the last convolutional block
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Pick a test image
img_path = "data/test/scratches/your_image.bmp"  # change this
img_pil  = Image.open(img_path).convert("RGB").resize((224, 224))
img_np   = np.array(img_pil) / 255.0

input_tensor = val_tf(Image.open(img_path)).unsqueeze(0).to(DEVICE)

grayscale_cam = cam(input_tensor=input_tensor)[0]
overlay = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_np); axes[0].set_title("Original")
axes[1].imshow(grayscale_cam, cmap='jet'); axes[1].set_title("Grad-CAM Heatmap")
axes[2].imshow(overlay); axes[2].set_title("Overlay")
for ax in axes: ax.axis('off')
plt.savefig("results/gradcam_example.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## Step 8: Anomaly Detection (Optional but Impressive)

Trains on GOOD images only. No need to label defects. Shows AI/ML depth in your project.

```python
# 04_anomaly.ipynb
# Uses Anomalib — handles everything for you

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path
import shutil

# Anomalib needs a specific folder structure:
# anomaly_data/
#   train/good/      ← only good images
#   test/good/       ← good images for testing
#   test/defective/  ← defective images for testing

# Create this structure from your existing data
def setup_anomalib_data(src_train_dir, src_test_dir, output_dir):
    # Combine all non-defective... 
    # For NEU: we treat all 6 classes as "defective" and create synthetic "good" 
    # OR better: use Casting dataset which has explicit good/defective split
    
    good_dir  = Path(output_dir) / "train" / "good"
    bad_dir   = Path(output_dir) / "test"  / "defective"
    test_good = Path(output_dir) / "test"  / "good"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)
    test_good.mkdir(parents=True, exist_ok=True)
    
    # Copy good samples to train/good
    for img in Path(src_train_dir, "good").glob("*"):
        shutil.copy(img, good_dir)
    
    # Copy defective samples to test/defective
    for cls in ["defective"]:
        for img in Path(src_test_dir, cls).glob("*"):
            shutil.copy(img, bad_dir)
    print("Anomalib data structure created!")

# > If using NEU (which has no "good" class), use the Casting dataset instead
# > Casting dataset on Kaggle has "ok_front" (good) and "def_front" (defective)

# Train PatchCore
datamodule = Folder(
    name="casting_defects",
    root="anomaly_data/",
    normal_dir="train/good",
    abnormal_dir="test/defective",
    normal_test_dir="test/good",
    task="classification",
    image_size=(224, 224),
    train_batch_size=16,
    eval_batch_size=16,
)

model = Patchcore(
    backbone="wide_resnet50_2",
    pre_trained=True,
    layers=["layer2", "layer3"],
    num_neighbors=9,
)

engine = Engine(max_epochs=1, accelerator="auto")
engine.fit(model=model, datamodule=datamodule)
results = engine.test(model=model, datamodule=datamodule)
print("AUROC:", results[0]['image_AUROC'])
# Expected: 0.93-0.98
```

---

## Step 9: Prediction Script (Single Image)

```python
# predict.py — run this from command line or import into the dashboard
import sys
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 
               'pitted_surface', 'rolled-in_scale', 'scratches']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint="models/efficientnet_best.pth"):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

def predict(image_path: str, model):
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path)
    tensor = tf(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    
    predicted_class = CLASS_NAMES[probs.argmax().item()]
    confidence = probs.max().item()
    
    # Simple pass/fail: anything detected = FAIL
    verdict = "FAIL" if confidence > 0.7 else "UNCERTAIN"
    
    return {
        "verdict": verdict,
        "defect_type": predicted_class,
        "confidence": round(confidence, 3),
        "all_probs": {c: round(p.item(), 3) for c, p in zip(CLASS_NAMES, probs)}
    }

if __name__ == "__main__":
    model = load_model()
    result = predict(sys.argv[1], model)
    print(result)
```

---

## Step 10: Streamlit Dashboard

This is your demo — upload an image, see the result.

```python
# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from predict import load_model, predict
import os, json
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="QC Inspection System", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .verdict-pass { color: #00c853; font-size: 2em; font-weight: bold; }
    .verdict-fail { color: #ff1744; font-size: 2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Load model (cached so it doesn't reload on every interaction)
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Layout ───
st.title("🔍 AI Quality Inspection System")
st.caption("Upload a product image to detect defects")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🚀 Run Inspection", type="primary"):
            with st.spinner("Analyzing..."):
                # Save temp file
                temp_path = f"results/temp_{uploaded.name}"
                os.makedirs("results", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                
                result = predict(temp_path, model)
            
            # Show verdict
            st.divider()
            if result['verdict'] == 'FAIL':
                st.markdown(f"<div class='verdict-fail'>❌ FAIL</div>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='verdict-pass'>✅ PASS</div>", 
                           unsafe_allow_html=True)
            
            st.metric("Defect Type", result['defect_type'])
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            # Probability bar chart
            probs_df = pd.DataFrame({
                "Defect": list(result['all_probs'].keys()),
                "Probability": list(result['all_probs'].values())
            }).sort_values("Probability", ascending=True)
            
            st.bar_chart(probs_df.set_index("Defect"))
            
            # Log to history
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "image": uploaded.name,
                "verdict": result['verdict'],
                "defect": result['defect_type'],
                "confidence": result['confidence']
            })

with col2:
    st.subheader("📋 Inspection History (This Session)")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Summary metrics
        total = len(df)
        fails = (df['verdict'] == 'FAIL').sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Inspected", total)
        m2.metric("Defects Found", fails)
        m3.metric("Pass Rate", f"{(total-fails)/total*100:.0f}%")
        
        # Color-coded table
        def highlight(row):
            color = "#2d0a0a" if row['verdict'] == 'FAIL' else "#0a2d0a"
            return [f"background-color: {color}"] * len(row)
        
        st.dataframe(
            df.style.apply(highlight, axis=1), 
            use_container_width=True
        )
        
        # Defect type distribution
        if fails > 0:
            defect_counts = df[df['verdict']=='FAIL']['defect'].value_counts()
            st.bar_chart(defect_counts)
        
        # Export to CSV
        if st.button("💾 Download Results as CSV"):
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download", csv, "inspection_results.csv", "text/csv")
    else:
        st.info("Upload and inspect some images to see history here.")

# Run with: streamlit run app.py
```

---

## How to Run Everything

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Download and split dataset
python split_dataset.py

# 3. Run notebooks in order (Jupyter)
jupyter notebook

# 4. After training, launch the dashboard
streamlit run app.py

# 5. Open browser at http://localhost:8501
```

---

## What to Include in Your Report

| Section | What to write |
|---|---|
| **Introduction** | Manual QC problems, what your system does |
| **Dataset** | NEU dataset, 6 classes, image samples, distribution chart |
| **Preprocessing** | CLAHE explanation + before/after images |
| **Classical CV** | Edge detection + GLCM features + Random Forest, report accuracy |
| **CNN Model** | EfficientNet-B0, transfer learning explanation, training curves, accuracy |
| **Anomaly Detection** | PatchCore concept (train on good only), AUROC score |
| **Results** | Comparison table: Classical vs CNN vs Anomaly detection |
| **Explainability** | Grad-CAM images showing which region caused detection |
| **Dashboard** | Screenshots of Streamlit app |
| **Conclusion** | What worked, what didn't, what you'd improve |

---

## Comparison Table to Put in Report

| Approach | Accuracy | Training Time | Labels Needed | Best For |
|---|---|---|---|---|
| Classical CV (edges + GLCM + RF) | ~72-78% | < 2 min | Image-level only | Baseline, explainable |
| EfficientNet-B0 (fine-tuned) | ~93-97% | 30-60 min | Image-level only | Main model |
| PatchCore (anomaly detection) | AUROC ~0.95+ | < 5 min | **None** (good images only) | Unknown defect types |

---

## requirements.txt

```
torch
torchvision
opencv-python
scikit-learn
scikit-image
matplotlib
seaborn
streamlit
albumentations
anomalib
grad-cam
pandas
pillow
tqdm
jupyter
```

---

> **Total estimated time**: 1 weekend to get everything working. Use Google Colab for the CNN training part if your laptop is slow (free GPU, just upload your data).
