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

## 📊 Model Evaluation

The model is evaluated using:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **Confusion Matrix**  

Evaluation is performed on validation data using `sklearn`.

---

## 🧠 Decision System

Prediction results are interpreted using confidence thresholds:

| Confidence Level | Decision |
|-----------------|---------|
| High            | Accept Prediction |
| Medium          | Needs Review |
| Low             | Uncertain |

### Special Handling
- **Ambulance class uses a lower threshold** to prioritize detection of emergency vehicles.

---

## 🌐 Web Application

### Features

- Image Upload Classification  
- Live Camera Detection  
- Real-time Prediction Display  
- Confidence Visualization  
- Audio Feedback:
  - Speech output for predictions  
  - Siren alert for ambulance detection  

---

## 📊 Visualization

- Training vs Validation Accuracy graph  
- Real-time evaluation metrics display  
- Confusion matrix representation  

---

## 🖥 Project Structure

```
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
```

---

## ⚙️ Setup Instructions

### Install Dependencies

```bash
pip install tensorflow flask numpy scikit-learn
```

---

### Train Model

```bash
python train.py
```

---

### Evaluate Model

```bash
python evaluate.py
```

---

### Run Application

```bash
python app.py
```

---

### Open in Browser

```
http://127.0.0.1:5000/
```

---

## ⚠️ Observations

- Some visual similarity between truck and bus may cause confusion  
- Model performance improves with more diverse training data  

---

## 👥 Team

**Tamil Geeks**

- Dhanush  
- Maha Mithra  
- Mohan Babu  
- Raksha  

---

## 📎 Notes

- Performance depends on dataset quality and diversity  
- The system is designed to demonstrate real-world classification behavior with decision interpretation  

---
