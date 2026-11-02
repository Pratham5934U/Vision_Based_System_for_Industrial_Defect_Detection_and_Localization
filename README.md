#  AI-Powered Industrial Surface Defect Detection

> Automated quality inspection using Classical CV, Deep Learning (EfficientNet-B0), and Unsupervised Anomaly Detection (PatchCore) — with a Streamlit web dashboard.

---

##  Project Overview

This system detects surface defects in industrial parts (steel, castings) using three complementary approaches:

| Approach | Accuracy | Training Time | Labels Needed |
|---|---|---|---|
| Classical CV (GLCM + Random Forest) | ~72–78% | < 2 min | Image-level |
| EfficientNet-B0 (fine-tuned CNN) | ~93–97% | 30–60 min | Image-level |
| PatchCore (Anomaly Detection) | AUROC ~0.95+ | < 5 min | **None** |

---

##  Project Structure

```
industrial-defect-detection/
│
├── data/                        ← dataset images (not tracked in git)
│   ├── train/
│   │   ├── crazing/
│   │   ├── inclusion/
│   │   ├── patches/
│   │   ├── pitted_surface/
│   │   ├── rolled-in_scale/
│   │   └── scratches/
│   └── test/
│       └── (same structure)
│
├── models/                      ← saved model weights (.pth)
├── results/                     ← output images, CSV logs
│
├── notebooks/
│   ├── 01_eda.py                ← Exploratory Data Analysis
│   ├── 02_classical_cv.py       ← Classical CV Baseline (GLCM + RF)
│   ├── 03_cnn_classifier.py     ← EfficientNet-B0 fine-tuning + Grad-CAM
│   └── 04_anomaly.py            ← PatchCore Anomaly Detection
│
├── app.py                       ← Streamlit dashboard
├── predict.py                   ← Single-image prediction script
├── split_dataset.py             ← Train/test split utility
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Set Up Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **NEU Surface Defect Database** from Kaggle:
-> https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Place it in the `data/` folder, then run:

```bash
python split_dataset.py
```

### 3. Run Notebooks (in order)

```bash
jupyter notebook
# Open notebooks/ and run 01 → 02 → 03 → 04
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

Open browser at **http://localhost:8501**

---

##  Dataset

**NEU Surface Defect Database** — 6 classes, 1800 images total (300 per class):
- `crazing` — network of fine cracks
- `inclusion` — embedded foreign particles
- `patches` — uneven surface patches
- `pitted_surface` — pits/holes on surface
- `rolled-in_scale` — oxide scale rolled into surface
- `scratches` — linear scratches

---

##  Methodology

```
Raw Image
   │
   ▼
[Preprocessing] ── CLAHE contrast enhancement + data augmentation
   │
   ├── [Classical CV] ── Canny edges + GLCM texture → Random Forest → ~75% acc
   │
   ├── [Deep Learning] ── EfficientNet-B0 fine-tune → Grad-CAM heatmap → ~95% acc
   │
   └── [Anomaly Detection] ── PatchCore (train on good only) → AUROC ~0.96
```

---

##  Results

Results (accuracy plots, confusion matrices, Grad-CAM overlays) are saved in the `results/` folder after running the notebooks.

---

##  Team

**Course:** Machine Learning + Fundamentals of Computer Vision (SEM 5)

---

##  References

1. Haralick et al., "Textural Features for Image Classification," IEEE 1973
2. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
3. Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019
4. Song & Yan, "Surface Defect Database for Hot-Rolled Steel Strip," 2013
5. Roth et al., "Towards Total Recall in Industrial Anomaly Detection (PatchCore)," CVPR 2022
6. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017
