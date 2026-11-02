# 📚 All Concepts Used — AI Quality Inspection Project

> **GPU**: NVIDIA RTX 4060 (8GB VRAM) — you're well equipped. Training times below reflect this.

---

## 🗺️ Concept Map Overview

```
AI Quality Inspection
│
├── 1. Computer Vision (Classical)
│   ├── Image representation & color spaces
│   ├── Histogram & CLAHE
│   ├── Gaussian blur / denoising
│   ├── Thresholding (Otsu's)
│   ├── Edge detection (Canny)
│   ├── Morphological operations
│   ├── Contour detection
│   └── GLCM texture features
│
├── 2. Machine Learning (Classical)
│   ├── Feature engineering
│   ├── Feature scaling (StandardScaler)
│   ├── Random Forest classifier
│   ├── Train/val/test splits
│   ├── Confusion matrix
│   ├── Precision, Recall, F1, Accuracy
│   └── ROC-AUC
│
├── 3. Deep Learning
│   ├── Convolutional Neural Networks (CNNs)
│   ├── Transfer Learning & Fine-tuning
│   ├── EfficientNet-B0 architecture
│   ├── Batch Normalization
│   ├── Dropout (regularization)
│   ├── Softmax & Cross-entropy loss
│   ├── Focal Loss (class imbalance)
│   ├── AdamW optimizer
│   ├── Learning rate scheduling (Cosine Annealing)
│   ├── Data augmentation
│   └── Early stopping
│
├── 4. Object Detection (Optional / Extension)
│   ├── Anchor-free detection (YOLO architecture)
│   ├── Bounding box regression
│   ├── Non-Maximum Suppression (NMS)
│   └── mAP (mean Average Precision)
│
├── 5. Anomaly Detection
│   ├── The concept of anomaly vs. novelty
│   ├── PatchCore algorithm
│   ├── Feature embeddings (pretrained backbone)
│   ├── Coreset subsampling
│   ├── k-Nearest Neighbors (kNN) scoring
│   └── AUROC metric
│
├── 6. Explainability (XAI)
│   ├── Gradient-weighted Class Activation Maps (Grad-CAM)
│   ├── Backpropagation through layers
│   └── Heatmap overlay visualization
│
├── 7. Data Engineering
│   ├── Dataset EDA (Exploratory Data Analysis)
│   ├── Class imbalance
│   ├── Train/val/test stratified split
│   └── Albumentations augmentation pipeline
│
└── 8. Deployment / Dashboard
    ├── Streamlit web app
    ├── CSV logging
    └── Plotly charts
```

---

## 1. 🖼️ Computer Vision — Classical

### 1.1 Image Representation
- **What it is**: Images are 2D (grayscale) or 3D (H × W × C) arrays of pixel values 0–255.
- **Why we use it**: All processing starts here — understanding pixel math is foundational.
- **In your project**: Loading images with OpenCV, inspecting dimensions.
- **Difficulty**: ⭐☆☆☆☆
- **Learn time**: Already know it / 30 min

---

### 1.2 Color Spaces (BGR, Grayscale, HSV)
- **What it is**: Different ways to represent color. OpenCV reads images as BGR by default. Grayscale removes color, keeping only brightness.
- **Why we use it**: NEU dataset is grayscale. Converting color images to grayscale reduces compute.
- **In your project**: `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
- **Difficulty**: ⭐☆☆☆☆
- **Learn time**: 15 min

---

### 1.3 Histogram & Histogram Equalization
- **What it is**: A histogram counts how many pixels have each brightness value (0–255). Equalization spreads the distribution to improve contrast.
- **Why we use it**: Defects on uniform surfaces are often low-contrast — equalization makes them visible.
- **In your project**: Foundation concept for CLAHE below.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

### 1.4 CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **What it is**: Like histogram equalization, but done locally in tiles of the image + clips extreme values to avoid noise amplification.
- **Why we use it**: Best preprocessing step for subtle surface defects. Industry standard.
- **In your project**: `cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(img)` — applied to every image before feeding to models.
- **Key parameter**: `clipLimit` controls how aggressive the enhancement is (4.0 is a good default).
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

### 1.5 Gaussian Blur / Denoising
- **What it is**: Smooths an image by averaging each pixel with its neighbors, weighted by a Gaussian curve. Reduces random noise.
- **Why we use it**: Camera noise can look like small defects. Denoising before edge detection prevents false positives.
- **In your project**: `cv2.GaussianBlur(img, (5,5), 0)` before Canny edge detection.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 30 min

---

### 1.6 Thresholding (Otsu's Method)
- **What it is**: Converts a grayscale image to binary (black/white) by picking a threshold value. Otsu's automatically finds the optimal threshold by minimizing intra-class variance.
- **Why we use it**: Separates the product (foreground) from the background for ROI extraction.
- **In your project**: `cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

### 1.7 Canny Edge Detection
- **What it is**: A multi-stage algorithm: Gaussian smoothing → gradient computation → non-maximum suppression → hysteresis thresholding. Finds edges (sharp intensity changes) in images.
- **Why we use it**: Defects like scratches and cracks create strong edges. Edge density is a useful classical feature.
- **In your project**: `cv2.Canny(img, low_threshold=50, high_threshold=150)`
- **Key parameters**: `low` and `high` thresholds — tune based on your images.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 2 hours

---

### 1.8 Morphological Operations (Erosion, Dilation, Opening, Closing)
- **What it is**: Operations that process image shapes using a structuring element (kernel). Opening removes small noise. Closing fills small holes.
- **Why we use it**: After thresholding, there's always noise. Morphological ops clean the binary mask before contour detection.
- **In your project**: `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)`
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

### 1.9 Contour Detection
- **What it is**: Finding the boundaries of connected regions in a binary image.
- **Why we use it**: Used to find the largest blob in the image (the product itself) and crop to it for ROI extraction.
- **In your project**: `cv2.findContours()` → `cv2.boundingRect()` to get crop coordinates.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

### 1.10 GLCM — Gray Level Co-occurrence Matrix
- **What it is**: A statistical method that captures texture by counting how often pairs of pixel values appear at a specific offset/angle. Produces features: contrast, homogeneity, energy, correlation, dissimilarity.
- **Why we use it**: Surface texture changes when there's a defect (e.g., a scratch makes the surface locally more "rough"). GLCM captures this.
- **In your project**: `skimage.feature.graycomatrix()` and `graycoprops()` — used as features for the Random Forest classifier.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 2–3 hours

---

## 2. 🤖 Machine Learning — Classical

### 2.1 Feature Engineering
- **What it is**: Manually extracting meaningful numeric representations from raw data (images) to feed into classical ML models.
- **Why we use it**: Random Forest can't take raw pixels (too many). You hand-craft meaningful features: edge density, GLCM values, histogram stats.
- **In your project**: Building the feature vector `[edge_density, std_dev, contrast, homogeneity, ...]` per image.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 2–4 hours

---

### 2.2 Feature Scaling (StandardScaler)
- **What it is**: Normalizes each feature to have zero mean and unit variance. Prevents features with large numeric ranges from dominating others.
- **Why we use it**: Your features have different scales (edge density is 0–1, std_dev is 0–255). Scaling puts them on equal footing.
- **In your project**: `sklearn.preprocessing.StandardScaler().fit_transform(X_train)`
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 30 min

---

### 2.3 Random Forest Classifier
- **What it is**: An ensemble of many decision trees. Each tree is trained on a random subset of data and features. Final prediction is a majority vote.
- **Why we use it**: Handles noisy features well, doesn't need much tuning, gives feature importance scores, runs fast.
- **In your project**: The classical CV baseline model — takes your hand-crafted features and classifies defect type.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 2–3 hours

---

### 2.4 Train / Validation / Test Split
- **What it is**: Dividing your dataset into three non-overlapping subsets: train (learn from), validation (tune hyperparameters), test (final unbiased evaluation).
- **Why we use it**: If you evaluate on training data, you get artificially high accuracy (overfitting). Test set is "untouched" until the very end.
- **In your project**: 70% train, 15% val, 15% test — stratified (each class proportion preserved).
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 30 min

---

### 2.5 Evaluation Metrics

| Metric | Formula | What it means |
|---|---|---|
| **Accuracy** | (TP+TN) / Total | Overall correct rate — misleading with imbalanced classes |
| **Precision** | TP / (TP+FP) | Of things flagged as defective, how many actually were? |
| **Recall** | TP / (TP+FN) | Of all actual defects, how many did we catch? **(Most important for QC)** |
| **F1 Score** | 2 × P×R / (P+R) | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under ROC curve | Overall ranking ability, 1.0 = perfect, 0.5 = random |
| **mAP** | Mean Avg. Precision | For object detection — measures detection + localization quality |
| **AUROC** | Same as ROC-AUC | Used specifically for anomaly detection |

- **In your project**: You report all of these in your Results section.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 2–3 hours

---

### 2.6 Confusion Matrix
- **What it is**: A table showing TP, FP, TN, FN for each class. Rows = actual, columns = predicted.
- **Why we use it**: Shows exactly which classes get confused with each other (e.g., model confuses `crazing` with `patches`).
- **In your project**: `sklearn.metrics.confusion_matrix()` + plotted with `ConfusionMatrixDisplay`.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

## 3. 🧠 Deep Learning

### 3.1 Convolutional Neural Networks (CNNs)
- **What it is**: A neural network that uses convolutional layers (learnable filters) to automatically extract spatial features from images — edges → shapes → parts → objects.
- **Why we use it**: CNNs learn better features than hand-crafted GLCM/edge density. The core of modern image recognition.
- **Key layers**: Conv2D → BatchNorm → ReLU → MaxPool → repeat → GlobalAvgPool → Linear
- **In your project**: EfficientNet-B0 is a CNN. Understanding it is essential.
- **Difficulty**: ⭐⭐⭐⭐☆
- **Learn time**: 5–10 hours (to understand well)

---

### 3.2 Transfer Learning & Fine-tuning
- **What it is**: Using a model pre-trained on a large dataset (ImageNet — 1.2M images, 1000 classes) as a starting point, then adapting it to your task.
- **Why we use it**: Training a full CNN from scratch on 1800 images would overfit badly. Pretrained weights already know about edges, textures, shapes — you just specialize the last layers.
- **Two strategies**:
  - **Feature extraction**: Freeze all layers, only train the new classification head (faster).
  - **Fine-tuning**: Unfreeze last few layers + head and train with a small learning rate (better accuracy).
- **In your project**: Fine-tune last 2 blocks + classifier of EfficientNet-B0.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 3–4 hours

---

### 3.3 EfficientNet-B0 Architecture
- **What it is**: A CNN architecture that scales width, depth, and resolution together using a compound coefficient. B0 is the smallest version — 5.3M parameters, very efficient.
- **Why we use it**: State-of-the-art accuracy at low compute cost. Runs well on RTX 4060.
- **Key innovation**: NAS (Neural Architecture Search) found the optimal layer configuration.
- **In your project**: Main defect classifier model.
- **Difficulty**: ⭐⭐⭐☆☆ (to use; ⭐⭐⭐⭐⭐ to understand fully)
- **Learn time**: 2–3 hours to understand the concept

---

### 3.4 Batch Normalization
- **What it is**: Normalizes layer activations within each mini-batch. Keeps values in a good range so gradients don't explode or vanish.
- **Why we use it**: Enables faster, more stable training. Used inside EfficientNet — you don't implement it yourself.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 1–2 hours to understand why it works

---

### 3.5 Dropout (Regularization)
- **What it is**: During training, randomly "drops" (zeros out) a fraction of neurons. Forces the network to not rely on any single neuron.
- **Why we use it**: Prevents overfitting on small datasets (like NEU with 1800 images).
- **In your project**: `nn.Dropout(0.3)` in the classifier head.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 30 min

---

### 3.6 Softmax & Cross-Entropy Loss
- **What it is**: Softmax converts raw model outputs (logits) into probabilities that sum to 1. Cross-entropy loss measures how far those probabilities are from the true one-hot labels.
- **Why we use it**: Standard loss function for multi-class classification.
- **In your project**: `nn.CrossEntropyLoss()` — PyTorch combines softmax + cross-entropy in one for numerical stability.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 2 hours

---

### 3.7 Focal Loss
- **What it is**: A modification of cross-entropy that reduces the weight of easy, correctly-classified examples and focuses training on hard, misclassified ones. Controlled by `gamma` parameter.
- **Why we use it**: When classes are imbalanced (many "good" parts, few defective), the model learns to always predict "good" because it still gets low loss. Focal loss fixes this.
- **Formula**: `FL = -α(1-pₜ)^γ log(pₜ)` — the `(1-pₜ)^γ` term downweights easy examples.
- **In your project**: Optional — use if you simulate real-world imbalance.
- **Difficulty**: ⭐⭐⭐⭐☆
- **Learn time**: 2–3 hours

---

### 3.8 AdamW Optimizer
- **What it is**: Adam optimizer (adaptive learning rates per parameter) + Weight Decay decoupled from the gradient update. The "W" stands for the proper weight decay fix.
- **Why we use it**: Best general-purpose optimizer for fine-tuning. Converges faster than SGD.
- **In your project**: `torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)`
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 1–2 hours

---

### 3.9 Learning Rate Scheduling
- **What it is**: Changing the learning rate during training. High LR at start (explore), low LR at end (fine-tune).
- **Cosine Annealing**: LR decreases following a cosine curve — smooth, no abrupt drops.
- **In your project**: `CosineAnnealingLR(optimizer, T_max=epochs)` — LR starts at 1e-4, decays to ~1e-6.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 1 hour

---

### 3.10 Data Augmentation
- **What it is**: Creating modified copies of training images (flips, rotations, brightness changes, noise) to artificially expand the dataset and improve generalization.
- **Why we use it**: 1800 images is small. Augmentation effectively multiplies your data 5–10×.
- **Library**: Albumentations (faster than torchvision transforms for complex augmentations).
- **In your project**: Applied only to training data. Never to validation/test data.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1–2 hours

---

### 3.11 Early Stopping
- **What it is**: Stop training when validation accuracy stops improving for N consecutive epochs (patience). Saves the best checkpoint.
- **Why we use it**: Prevents wasting GPU time on epochs that don't help (or hurt via overfitting).
- **In your project**: Monitor `val_accuracy`, save `best.pth` when it improves.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 30 min

---

## 4. 🎯 Object Detection (Optional Extension)

### 4.1 YOLO Architecture (YOLOv8)
- **What it is**: "You Only Look Once" — detects and classifies multiple objects in a single forward pass. Divides image into grid cells, each predicts bounding boxes + class probabilities.
- **Why we use it**: When you need to not just classify but *locate* defects with a bounding box (where exactly is the scratch?).
- **In your project**: Extension — train if you have XML annotation files with bounding boxes.
- **Difficulty**: ⭐⭐⭐⭐☆
- **Learn time**: 4–5 hours

---

### 4.2 Bounding Box Regression
- **What it is**: Predicting 4 values (x, y, width, height) to localize an object in an image.
- **In your project**: Output of YOLOv8 per detection.
- **Difficulty**: ⭐⭐⭐☆☆

---

### 4.3 mAP (mean Average Precision)
- **What it is**: Evaluation metric for object detection. Computes precision-recall area under curve at different IoU thresholds, averaged over all classes.
- **Key variants**: mAP@0.5 (box must overlap ground truth ≥ 50%), mAP@0.5:0.95 (stricter).
- **In your project**: The main metric when evaluating YOLOv8.
- **Difficulty**: ⭐⭐⭐⭐☆
- **Learn time**: 2–3 hours

---

## 5. 🔬 Anomaly Detection

### 5.1 Anomaly Detection vs Classification
- **What it is**: Classification learns from examples of each class. Anomaly detection learns only what "normal" looks like — anything too different is flagged as anomalous.
- **Why it matters**: In real QC, you may not have many defect samples, or new defect types appear. Anomaly detection handles this naturally.
- **In your project**: Train PatchCore ONLY on "good" images.
- **Difficulty**: ⭐⭐⭐☆☆ (concept); ⭐⭐☆☆☆ (using Anomalib library)
- **Learn time**: 2 hours

---

### 5.2 Feature Embeddings
- **What it is**: The internal vector representation of an image inside a neural network (output of an intermediate layer). Two similar images will have similar embeddings.
- **Why we use it**: Instead of training a new model, use a pretrained network (WideResNet-50) as a feature extractor. Its embeddings capture rich visual information.
- **In your project**: PatchCore extracts patch-level embeddings from `layer2` and `layer3` of WideResNet-50.
- **Difficulty**: ⭐⭐⭐⭐☆
- **Learn time**: 2–3 hours

---

### 5.3 PatchCore Algorithm
- **What it is**: 
  1. Extract patch features from pretrained backbone at multiple scales.
  2. Build a **memory bank** of all patch features from training (good) images.
  3. Apply **coreset subsampling** — keep only the most representative 1% of patches (so the memory bank doesn't get too large).
  4. At test time: compute distance from each test patch to its nearest neighbor in the memory bank. High distance = anomaly.
  5. Aggregate patch scores into an image-level anomaly score.
- **Why use it**: State-of-the-art, no training (no gradient updates), trains in minutes, AUROC ~0.95+.
- **In your project**: `anomalib.models.Patchcore` — the library handles all of this.
- **Difficulty**: ⭐⭐⭐⭐☆ (concept); ⭐⭐☆☆☆ (using Anomalib)
- **Learn time**: 3–4 hours (to understand the paper)

---

### 5.4 Coreset Subsampling
- **What it is**: A technique to select a representative subset of N points from a large dataset such that the subset approximates the full set as well as possible. Uses greedy farthest-point selection.
- **Why we use it**: You'd have millions of patch embeddings — too much RAM. Coreset keeps ~1% that still covers the full distribution.
- **Difficulty**: ⭐⭐⭐⭐☆
- **Learn time**: 1 hour

---

### 5.5 AUROC (Area Under ROC Curve)
- **What it is**: Measures how well an anomaly detector separates normal from anomalous samples at all possible thresholds. 1.0 = perfect, 0.5 = random.
- **Why use AUROC instead of accuracy**: Anomaly detection doesn't have a fixed threshold — AUROC evaluates the ranking ability independent of threshold.
- **In your project**: Primary metric for PatchCore evaluation.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 1–2 hours

---

## 6. 🔍 Explainability (XAI)

### 6.1 Grad-CAM (Gradient-weighted Class Activation Mapping)
- **What it is**: 
  1. Forward pass: get prediction.
  2. Backward pass: compute gradients of the target class score with respect to the last convolutional layer's feature maps.
  3. Average gradients spatially (Global Average Pooling) → importance weights per channel.
  4. Weighted sum of feature maps → heatmap showing which spatial regions mattered.
  5. Overlay on original image with a color map.
- **Why we use it**: Shows WHERE in the image the model detected the defect. Critical for trust — professor/examiner can visually verify the model is looking at the right place.
- **In your project**: `pytorch-grad-cam` library. One call generates the heatmap.
- **Output**: Red/yellow = model focused here. Blue = irrelevant region.
- **Difficulty**: ⭐⭐⭐⭐☆ (concept); ⭐⭐☆☆☆ (using the library)
- **Learn time**: 2–3 hours

---

## 7. 📊 Data Engineering

### 7.1 Exploratory Data Analysis (EDA)
- **What it is**: Visually and statistically exploring your dataset before building models.
- **What to check**: Class distribution, image sizes, sample images, pixel value distribution, any corrupt files.
- **In your project**: Notebook `01_eda.ipynb`.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1–2 hours

---

### 7.2 Class Imbalance
- **What it is**: When some classes have many more samples than others. Classifiers tend to be biased toward majority classes.
- **NEU dataset**: Balanced (300 per class) — but in real deployment, good parts vastly outnumber defective ones.
- **Solutions**: Oversampling (duplicate minority), undersampling, class weights, Focal Loss, augmentation.
- **In your project**: Mention this as a challenge in your report even if NEU is balanced.
- **Difficulty**: ⭐⭐⭐☆☆
- **Learn time**: 1–2 hours

---

### 7.3 Albumentations Augmentation Pipeline
- **What it is**: A fast library for image augmentation. More transforms than torchvision, hardware-accelerated.
- **Key transforms you'll use**: `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `GaussNoise`, `CLAHE`, `RandomBrightnessContrast`, `ElasticTransform`
- **In your project**: Applied to training data only.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1 hour

---

## 8. 🖥️ Deployment / Dashboard

### 8.1 Streamlit
- **What it is**: Python library that turns Python scripts into interactive web apps with zero HTML/CSS/JS.
- **Key components**: `st.file_uploader`, `st.image`, `st.metric`, `st.dataframe`, `st.bar_chart`, `st.button`
- **In your project**: `app.py` — your entire demo UI.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 2–3 hours

---

### 8.2 Plotly Charts
- **What it is**: Interactive charting library. Works inside Streamlit with `st.plotly_chart()`.
- **In your project**: Defect distribution pie chart, confidence histogram, defect rate timeline.
- **Difficulty**: ⭐⭐☆☆☆
- **Learn time**: 1–2 hours

---

### 8.3 CSV Logging
- **What it is**: Saving inspection results (image name, verdict, confidence, timestamp) to a `.csv` file with pandas.
- **In your project**: Simple `df.to_csv("results/log.csv")` — enables downloading history from the dashboard.
- **Difficulty**: ⭐☆☆☆☆
- **Learn time**: 15 min

---

## 📐 Project Size Estimation

| Component | Files | Approx. Lines of Code | Complexity |
|---|---|---|---|
| EDA notebook | 1 | ~80 | Low |
| Classical CV notebook | 1 | ~150 | Medium |
| CNN training notebook | 1 | ~200 | Medium-High |
| Anomaly detection notebook | 1 | ~80 | Low (Anomalib does the heavy lifting) |
| `predict.py` | 1 | ~60 | Low |
| `app.py` (Streamlit) | 1 | ~150 | Medium |
| Dataset split script | 1 | ~30 | Low |
| **Total** | **7 files** | **~750 lines** | **Medium** |

> **This is a medium-sized project.** Not a weekend hack, not a 3-month grind. Realistic for 3–5 weeks of focused evenings.

---

## ⏱️ Time & GPU Estimates (RTX 4060, NEU Dataset)

| Task | CPU Time | RTX 4060 Time |
|---|---|---|
| Classical CV (feature extraction + RF) | ~3–5 min | Same (no GPU needed) |
| EfficientNet-B0 training (30 epochs) | ~45–60 min | **3–5 min** ✅ |
| YOLOv8s training (100 epochs) | ~4–6 hours | **12–20 min** ✅ |
| PatchCore training | ~2–4 min | **1–2 min** ✅ |
| Grad-CAM generation (per image) | <1s | <0.5s |
| **Total training time** | **~6–8 hours** | **~25–30 min** 🚀 |

> Your RTX 4060 is excellent for this project. You'll train everything in under an hour total.

---

## 🔧 Libraries & What Each Does

| Library | Purpose | Used For |
|---|---|---|
| `torch` + `torchvision` | Deep learning framework | EfficientNet training, tensors, transforms |
| `opencv-python` | Computer vision | Image loading, CLAHE, Canny, contours |
| `scikit-learn` | Classical ML | Random Forest, StandardScaler, metrics |
| `scikit-image` | Image features | GLCM texture features |
| `albumentations` | Data augmentation | Training augmentation pipeline |
| `anomalib` | Anomaly detection | PatchCore — wraps the full pipeline |
| `grad-cam` | Explainability | Grad-CAM heatmap generation |
| `streamlit` | Dashboard UI | Web app for demo |
| `plotly` | Charts | Interactive visualizations in dashboard |
| `pandas` | Data manipulation | CSV logging, results dataframes |
| `matplotlib` + `seaborn` | Static plotting | Training curves, confusion matrices |
| `tqdm` | Progress bars | Loop progress during training |
| `Pillow` | Image I/O | Loading images in Streamlit |

---

## 📝 Concepts by Difficulty

### Easy (understand in < 1 hour each)
- Image representation, grayscale conversion
- Train/val/test split
- Accuracy, confusion matrix
- Dropout
- Early stopping
- CSV logging
- Streamlit basics

### Medium (1–3 hours each)
- CLAHE, Gaussian blur, thresholding
- Canny edge detection
- GLCM features
- Feature scaling, Random Forest
- Precision, Recall, F1, ROC-AUC
- Softmax, Cross-Entropy Loss
- AdamW, learning rate scheduling
- Transfer learning concept
- Data augmentation
- Anomaly detection concept

### Hard (3–6 hours each)
- CNNs — how convolutions work mathematically
- EfficientNet architecture in depth
- Focal Loss — why it works
- Grad-CAM — backpropagation understanding
- PatchCore — reading and understanding the paper
- mAP for object detection
- AUROC interpretation

---

## ✅ What You MUST Know vs What's Nice to Know

### Must Know (examiners will ask)
- [ ] What is CLAHE and why do you use it?
- [ ] What is transfer learning and why not train from scratch?
- [ ] What's the difference between precision and recall? Which is more important for QC?
- [ ] What is Grad-CAM and what does the heatmap show?
- [ ] What is anomaly detection? Why train only on good images?
- [ ] What is the confusion matrix showing?
- [ ] What is your best model's accuracy/F1/AUROC?

### Nice to Know (for deeper questions)
- [ ] How does Canny edge detection work internally?
- [ ] Why use AdamW over Adam?
- [ ] How does PatchCore's coreset subsampling work?
- [ ] Why does focal loss help with class imbalance?
- [ ] What is mAP and how is it calculated?
