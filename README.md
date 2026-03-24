# 🚗 Vehicle Classification AI System

A real-time deep learning application that classifies vehicles and evaluates prediction confidence with an integrated decision system and interactive web interface.

---

## 📌 Overview

This project is a complete end-to-end AI system that performs vehicle classification using a trained deep learning model.  
It supports both static image input and live camera detection, enhanced with confidence evaluation and audio feedback.

---

## 🎯 Objective

To build a reliable vehicle classification system that:
- Accurately predicts vehicle types
- Evaluates prediction confidence
- Demonstrates real-time AI interaction

---

## 🚘 Classes

- 🚗 Car  
- 🏍 Bike  
- 🚌 Bus  
- 🚛 Truck  
- 🚑 Ambulance  

---

## ⚙️ System Flow

1. Image Input (Upload / Live Camera)  
2. Preprocessing (resize + normalize)  
3. Feature Extraction (MobileNetV2)  
4. Classification (Softmax layer)  
5. Confidence Evaluation  
6. Result + Audio Output  

---

## 🌐 Application Features

### 🖼 Upload Mode
- Upload any image
- Instant classification
- Confidence score + decision

### 🎥 Live Camera Mode
- Uses webcam in real-time
- Capture frame and classify
- Immediate prediction display

### 🔊 Audio Feedback
- Voice output for detected class
- 🚨 Ambulance triggers siren sound

### 📊 Decision System
- High Confidence → Accept  
- Medium → Needs Review  
- Low → Uncertain  

---

## 🧠 Model Details

- Architecture: MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Custom classifier head added
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## 🧹 Preprocessing

- Resize images to **224 × 224**
- Normalize pixel values (0–1)
- Data augmentation:
  - Flip
  - Rotation
  - Zoom

---

## 📁 Dataset

- Sources: Kaggle, Zenodo
- Total Images: ~2000+
- Classes: 5

---

## 🧪 Evaluation (Real Model Results)

Evaluation performed using validation dataset and `sklearn`.

| Metric     | Value |
|-----------|------|
| Accuracy  | 91.16% |
| Precision | 91.53% |
| Recall    | 91.16% |

---

## 🔍 Confusion Matrix

| Actual \ Pred | ambulance | bike | bus | car | truck |
|---------------|----------|------|-----|-----|-------|
| ambulance     | 83 | 0 | 3 | 0 | 2 |
| bike          | 0 | 83 | 1 | 0 | 0 |
| bus           | 0 | 0 | 84 | 0 | 0 |
| car           | 0 | 0 | 3 | 78 | 9 |
| truck         | 6 | 0 | 11 | 3 | 64 |

---

## 📈 Key Observations

- Strong overall performance (~91% accuracy)
- Minor confusion between:
  - Truck ↔ Bus
  - Car ↔ Truck
- Ambulance detection is reliable

---

## 🖥 Project Structure

```
project/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── live.html
│   ├── result.html
│   ├── about.html
│   └── engineering.html
├── static/
│   ├── sounds/
│   └── data/
└── README.md
```

---

## 🚀 Running the Project

### Install dependencies

```
pip install tensorflow flask numpy scikit-learn
```

### Run directly (NO retraining required)

```
python app.py
```

### Open in browser

```
http://127.0.0.1:5000/
```

---

## ⚠️ Notes

- Model is already trained — no need to retrain
- Evaluation results are based on validation dataset
- Performance depends on dataset diversity

---

## 👥 Team

**Tamil Geeks**

- Dhanush  
- Maha Mithra  
- Mohan Babu  
- Raksha  

---

## 🏁 Conclusion

This system demonstrates a practical implementation of deep learning with real-time interaction, evaluation, and decision interpretation in a web-based environment.

---
