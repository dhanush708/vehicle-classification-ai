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

---

## 🧹 Preprocessing Steps

- Resize all images to **224 × 224**  
- Normalize pixel values (**0–1 scaling**)  
- Data augmentation (flip, rotation, zoom)  
- Dataset cleaning  

---

## 🧠 Model Architecture

- MobileNetV2 (Transfer Learning)  
- Global Average Pooling  
- Dense Layer  
- Softmax Output (5 classes)  

---

## 🧪 Evaluation Pipeline

Evaluation is performed using `evaluate.py` on validation data.

Metrics are computed using `sklearn`.

---

## 📊 Real Model Performance

The following results are obtained from actual evaluation:

| Metric     | Value |
|-----------|------|
| Accuracy  | 91.16% |
| Precision | 91.53% |
| Recall    | 91.16% |

---

## 🔍 Confusion Matrix (Actual)

| Actual \ Pred | ambulance | bike | bus | car | truck |
|---------------|----------|------|-----|-----|-------|
| ambulance     | 83 | 0 | 3 | 0 | 2 |
| bike          | 0 | 83 | 1 | 0 | 0 |
| bus           | 0 | 0 | 84 | 0 | 0 |
| car           | 0 | 0 | 3 | 78 | 9 |
| truck         | 6 | 0 | 11 | 3 | 64 |

---

## 📈 Interpretation

- Strong overall accuracy (~91%)  
- Minor confusion between **truck, bus, and car**  
- Ambulance detection is reliable  

---

## 🧠 Decision System

- High confidence → Accept  
- Medium → Needs Review  
- Low → Uncertain  
- Ambulance → prioritized  

---

## 🌐 Web Application Features

- Image Upload  
- Live Camera Detection  
- Real-time Prediction  
- Confidence Display  
- Audio Feedback (speech + siren)  

---

## 🖥 Project Structure

```
project/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── templates/
├── static/
└── README.md
```

---

## ⚙️ Setup

```
pip install tensorflow flask numpy scikit-learn
```

```
python train.py
python evaluate.py
python app.py
```

---

## 👥 Team

Tamil Geeks  
Dhanush, Maha Mithra, Mohan Babu, Raksha  

---

## 📎 Notes

- Metrics shown are from actual model evaluation  
- Performance depends on dataset quality  

---
