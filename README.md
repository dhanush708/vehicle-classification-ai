# 🚗 Vehicle Classification AI System

A deep learning-based web application that classifies vehicles and evaluates prediction confidence using a structured decision system.

---

## 📌 Overview

This project implements an end-to-end vehicle classification system using deep learning.  
It supports both static image input and real-time camera-based prediction, along with a confidence-based evaluation layer.

---

## 🎯 Objective

To accurately classify vehicles into predefined categories and assess prediction reliability using model confidence.

---

## 🚘 Classes Used

- 🚗 Car  
- 🏍 Bike  
- 🚌 Bus  
- 🚛 Truck  
- 🚑 Ambulance  

---

## ⚙️ System Workflow

1. Image Input (Upload / Live Camera)  
2. Preprocessing  
3. Feature Extraction (CNN)  
4. Classification (Softmax Output)  
5. Confidence Evaluation  
6. Result Display + Audio Feedback  

---

## 📁 Dataset

### Sources
- Kaggle  
- Zenodo  

### Details

- Total Images: ~2000+  
- Number of Classes: 5  

| Class       | Approx Images |
|------------|--------------|
| Car        | 300–400 |
| Bike       | 300–400 |
| Bus        | 300–400 |
| Truck      | 300–400 |
| Ambulance  | 500+ |

---

## 🖼 Image Characteristics

- Mixed resolutions (standardized during preprocessing)  
- Formats: JPG, PNG  
- Real-world variations in lighting and angles  

---

## 🧹 Preprocessing Steps

- Resize all images to **224 × 224**  
- Normalize pixel values (**0–255 → 0–1**)  
- Apply data augmentation:
  - Rotation  
  - Zoom  
  - Horizontal Flip  
- Remove noisy / incorrect samples  
- Split into training and validation datasets  

---

## 🧠 Model Architecture

### Base Model
- **MobileNetV2 (Pretrained on ImageNet)**

### Custom Layers
- Global Average Pooling  
- Dense Layer (ReLU)  
- Softmax Output Layer (5 classes)  

---

## 🔄 Training Strategy

- Transfer learning approach  
- Base layers frozen  
- Only top layers trained  
- Optimized using Adam optimizer  
- Loss function: Categorical Crossentropy  

---

## 🧪 Evaluation Pipeline

Model evaluation is performed using a separate script (`evaluate.py`) on the validation dataset.

### Steps

1. Load trained model  
2. Run predictions on validation data  
3. Compare predictions with true labels  
4. Compute metrics using `sklearn`  

### Metrics Computed

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- Confusion Matrix  

### Output File

static/data/metrics.json

This file is dynamically loaded in the Engineering page.

---

## 📊 Example Output Format

{
  "accuracy": 0.91,
  "precision": 0.90,
  "recall": 0.89,
  "classes": ["ambulance", "bike", "bus", "car", "truck"],
  "confusion_matrix": [[...]]
}

---

## 📈 Interpretation

- Accuracy → overall correctness  
- Precision → reliability of predictions  
- Recall → ability to detect each class  
- Confusion matrix → shows misclassification patterns  

---

## 🧠 Decision System

Prediction results are interpreted using confidence thresholds:

| Confidence Level | Decision |
|-----------------|---------|
| High            | Accept Prediction |
| Medium          | Needs Review |
| Low             | Uncertain |

### Special Handling
- Ambulance class uses a lower threshold for emergency priority  

---

## 🌐 Web Application

### Features

- Image Upload Classification  
- Live Camera Detection  
- Real-time Prediction Display  
- Confidence Visualization  
- Audio Feedback:
  - Speech output  
  - Ambulance siren alert  

---

## 📊 Visualization

- Training vs Validation Accuracy graph  
- Real-time evaluation metrics  
- Confusion matrix  

---

## 🖥 Project Structure

project/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── dataset/
├── model/
├── static/
│   └── data/
│       └── metrics.json
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── live.html
│   ├── result.html
│   ├── about.html
│   └── engineering.html
└── README.md

---

## ⚙️ Setup Instructions

Install dependencies:
pip install tensorflow flask numpy scikit-learn

Train model:
python train.py

Evaluate model:
python evaluate.py

Run app:
python app.py

Open:
http://127.0.0.1:5000/

---

## ⚠️ Observations

- Some confusion between visually similar classes (truck vs bus)  
- Performance improves with better dataset diversity  

---

## 👥 Team

Tamil Geeks

- Dhanush  
- Maha Mithra  
- Mohan Babu  
- Raksha  

---

## 📎 Notes

- Performance depends on dataset quality  
- Designed to demonstrate real-world ML behavior with decision logic  

---
