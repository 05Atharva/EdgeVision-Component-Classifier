# EdgeVision – Electronic Component Classifier

A lightweight **computer vision + TinyML-ready** image classification project for
distinguishing **Capacitors** and **Inductors** using transfer learning.

This repository currently focuses on **model training, evaluation, and image-based inference**.
Embedded deployment and automation are planned as future work.

---

## 🔍 Problem Statement

Manual segregation of small electronic components is time-consuming and error-prone.
This project explores how **edge-friendly deep learning models** can classify
electronic components using images, forming the foundation for future automation.

---

## 🧠 Model Overview

- Architecture: **MobileNetV2** (Transfer Learning)
- Trained Model File: `component_classifier_mobilenetv2.h5`
- Input Size: **96 × 96 RGB**
- Classes:
  - Capacitor
  - Inductor
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam (learning rate = 1e-4)
- Class Imbalance Handling: Class Weights

The model is designed to be lightweight and suitable for **future edge deployment**.

---

## 📊 Performance

- Validation Accuracy: **~99–100%**
- Confusion Matrix:

[[317 0]
[ 3 299]]


- Precision / Recall / F1-score: **≈ 1.00**

Detailed evaluation results are available in the `/results` directory.

---

## 🧪 Inference Example

Run inference on a folder of test images:

```bash
python inference/image_inference.py
con1.jfif → Capacitor ⚡ (Confidence: 0.99)
ind1.jfif → Inductor 🌀 (Confidence: 0.99)


## 🛠 Tech Stack

Python

TensorFlow / Keras

MobileNetV2

NumPy

Matplotlib

Scikit-learn

## 🚀 Future Scope

TensorFlow Lite (TFLite) model conversion

Deployment on ESP32-CAM

Conveyor-belt-based automated component segregation

Real-time edge inference and control logic

Extension to multi-class component classification

⚠️ PS (Important)

Note:
This repository currently contains only the trained ML model and inference pipeline.
Hardware integration (ESP32, camera module, conveyor belt) is planned future work and is
not part of the present implementation.

👤 Author

Atharva Kanawade
Electronics & Telecommunication | Machine Learning | Embedded Systems | Edge AI