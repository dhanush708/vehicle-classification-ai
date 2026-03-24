import tensorflow as tf
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model("model/vehicle_classifier.h5")

# Data path (IMPORTANT: use your validation folder)
val_dir = "dataset/val"

IMG_SIZE = (224, 224)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

class_labels = list(val_generator.class_indices.keys())

# Metrics
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
cm = confusion_matrix(y_true, y_pred).tolist()

# Save results
results = {
    "accuracy": report["accuracy"],
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "classes": class_labels,
    "confusion_matrix": cm
}

os.makedirs("static/data", exist_ok=True)

with open("static/data/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Evaluation saved to static/data/metrics.json")