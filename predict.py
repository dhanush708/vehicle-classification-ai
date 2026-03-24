import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("model/vehicle_classifier.h5")

# Class labels (IMPORTANT: must match training order)
class_names = ['ambulance', 'bike', 'bus', 'car', 'truck']

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dims
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    # Get class + confidence
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    return predicted_class, confidence

# Test
if __name__ == "__main__":
    img_path = r"C:\Users\DHANUSH ANBU\Downloads\bus.jpg"#(ambulance)
    label, conf = predict_image(img_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2f}")