# 🚗 Vehicle Classification AI System

A deep learning-based application that classifies vehicle images into multiple categories and evaluates prediction confidence using a structured decision system.

---

## 📌 Overview

This project focuses on building an image classification system capable of identifying vehicles into the following categories:

- 🚗 Car  
- 🏍 Bike  
- 🚛 Truck  
- 🚌 Bus  
- 🚑 Ambulance  

The system includes a prediction layer and a confidence-based evaluation mechanism.

---

## 🔍 Problem Statement

Classify vehicle images accurately and determine how reliable each prediction is based on model confidence.

---

## ⚙️ System Workflow

1. Image Upload  
2. Image Preprocessing  
3. Feature Extraction (CNN)  
4. Classification (Softmax Output)  
5. Confidence Evaluation  

---

## 📊 Dataset

### Sources
- Kaggle  
- Zenodo  

### Details

- Total Images: ~2000+  
- Number of Classes: 5  

| Class | Approx Images |
|------|--------------|
| Car | 350–400 |
| Bike | 350–400 |
| Truck | 350–400 |
| Bus | 300–400 |
| Ambulance | 500+ |

---

## 🖼 Image Characteristics

- Mixed resolutions (low to high)  
- Formats: JPG, PNG  
- Standardized during preprocessing  

---

## 🧹 Preprocessing Steps

- Resize all images to 224 × 224  
- Normalize pixel values (0–255 → 0–1)  
- Apply data augmentation:
  - Rotation  
  - Zoom  
  - Horizontal Flip  

- Split dataset into training and validation sets  
- Remove incorrect or noisy samples  

---

## 🧠 Model Architecture

### Base Model
- MobileNetV2 (pretrained on ImageNet)

### Custom Layers
- Global Average Pooling  
- Dense Layer (ReLU)  
- Output Layer (Softmax – multi-class classification)

---

## 🔄 Transfer Learning

- Pretrained weights reused  
- Base layers frozen  
- Only top layers trained  

---

## 📈 Training Process

- Model trained on prepared dataset  
- Validation used to monitor performance  
- Multiple epochs used for convergence  

---

## 📊 Evaluation

- Accuracy used as evaluation metric  
- Model performs consistently on validation data  

---

## ⚠️ Observations

- Some overlap between truck and bus due to similar shapes  
- Performance can improve with more varied data  

---

## 🧠 Decision Logic

The model output is interpreted using confidence thresholds:

- High confidence → Accept prediction  
- Medium confidence → Needs review  
- Low confidence → Uncertain  

### Special Case: Ambulance
- Lower threshold applied to reduce risk of missing emergency vehicles  

---

## 🌐 Web Application

### Features

- Image upload interface  
- Drag-and-drop support  
- Image preview  
- Separate result page  
- Confidence visualization  
- Emergency alert display  

---

## 🖥 Project Structure

```
project/
├── app.py
├── train.py
├── predict.py
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── about.html
│   └── engineering.html
├── static/
└── model/
```

---

## ⚙️ Setup Instructions

### Install Dependencies
```
pip install tensorflow flask numpy matplotlib
```

### Run Application
```
python app.py
```

### Open in Browser
```
http://127.0.0.1:5000/
```

---

## 👥 Team

Tamil Geeks  

- Dhanush  
- Maha Mithra  
- Mohan Babu  
- Raksha  

---

## 📎 Notes

- Model performance depends on dataset diversity  
- Additional data can further improve accuracy  
