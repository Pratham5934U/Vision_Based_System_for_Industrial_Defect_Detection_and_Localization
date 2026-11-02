# PROJECT SYNOPSIS

## 1. Title
**AI-Powered Automated Quality Inspection and Surface Defect Detection System**

---

## 2. Introduction
In manufacturing industries, inspecting products for surface defects (cracks, scratches, dents, inclusions) is crucial for quality assurance. Traditionally, this is done via manual human inspection, which is slow, inconsistent, and prone to fatigue. 

This project aims to build an automated visual inspection system using Computer Vision and Deep Learning. The system takes product surface images, preprocesses them to enhance subtle defects, classifies defect types using Convolutional Neural Networks (CNNs), detects unseen defects using unsupervised anomaly detection, and highlights defect regions using Grad-CAM heatmaps. A simple Streamlit web interface is developed for easy demonstration and test image verification.

---

## 3. Literature Review
1. **Classical Computer Vision:** Early automated inspection systems relied on handcrafted texture descriptors like GLCM (Haralick et al., 1973) and edge detection (Canny, Otsu). While fast, they struggle with low contrast and complex background textures.
2. **Deep Learning Classification:** Modern approaches use pretrained Convolutional Neural Networks such as ResNet (He et al., 2016) and EfficientNet (Tan & Le, 2019) to automatically learn spatial defect features, achieving high classification accuracy on benchmark datasets like NEU Surface Defects (Song & Yan, 2013).
3. **Unsupervised Anomaly Detection:** Since real-world defective samples are rare, models like PatchCore (Roth et al., 2022) train exclusively on normal/good images and detect anomalies based on feature embedding distance.
4. **Model Explainability:** Grad-CAM (Selvaraju et al., 2017) is widely used in visual inspection to generate visual heatmaps, confirming that the deep learning model focuses on the actual defect rather than background noise.

---

## 4. Research Gaps & Objectives

### Research Gaps
- **Rare Defect Handling:** Supervised models fail on new or unseen defect types not present during training.
- **Lack of Baseline Comparison:** Many projects only implement deep learning without comparing against classical CV baselines.
- **Black-Box Nature:** Standard models give a classification label without showing where the defect is located.

### Objectives
1. **Data Preprocessing:** Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) and apply data augmentation.
2. **Classical Baseline:** Build an edge- and texture-based (GLCM) Random Forest classifier as a baseline.
3. **Deep Learning Model:** Fine-tune an EfficientNet-B0 CNN to classify defect types with high recall ($\ge 95\%$).
4. **Anomaly Detection:** Implement PatchCore to identify defective parts trained only on non-defective images.
5. **Visual Explainability:** Use Grad-CAM to overlay heatmaps on defect regions.
6. **User Dashboard:** Build a simple Streamlit web app to upload images, view pass/fail predictions, and view inspection logs.

---

## 5. References
1. Haralick, R. M. et al., "Textural Features for Image Classification," *IEEE Trans. Syst., Man, Cybern.*, 1973.
2. He, K. et al., "Deep Residual Learning for Image Recognition," *IEEE CVPR*, 2016.
3. Tan, M., & Le, Q., "EfficientNet: Rethinking Model Scaling for CNNs," *ICML*, 2019.
4. Song, K., & Yan, Y., "Surface Defect Database for Hot-Rolled Steel Strip," *Applied Surface Science*, 2013.
5. Roth, K. et al., "Towards Total Recall in Industrial Anomaly Detection (PatchCore)," *IEEE CVPR*, 2022.
6. Selvaraju, R. R. et al., "Grad-CAM: Visual Explanations from Deep Networks," *IEEE ICCV*, 2017.
