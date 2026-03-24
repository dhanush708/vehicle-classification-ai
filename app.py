import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/vehicle_classifier.h5")

# Classes
class_names = ['ambulance', 'bike', 'bus', 'car', 'truck']


def get_decision(predicted_class, confidence):

    if predicted_class == "ambulance":
        if confidence >= 0.75:
            return "🚨 High Confidence (Emergency)"
        elif confidence >= 0.60:
            return "❓ Needs Review"
        else:
            return "⚠️ Uncertain"

    else:
        if confidence >= 0.85:
            return "✅ High Confidence"
        elif confidence >= 0.65:
            return "❓ Needs Review"
        else:
            return "⚠️ Uncertain"


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    confidence = float(np.max(predictions))
    predicted_class = class_names[np.argmax(predictions)]

    decision = get_decision(predicted_class, confidence)

    return predicted_class, confidence, decision


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/live")
def live():
    return render_template("live.html")


# 🔥 ADD THESE (THIS IS YOUR MISSING PART)

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/engineering")
def engineering():
    return render_template("engineering.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    file_path = "static/temp.jpg"
    file.save(file_path)

    label, confidence, decision = predict_image(file_path)

    result = {
        "class": label,
        "confidence": confidence,
        "decision": decision
    }

    return render_template("result.html",
                           result=result,
                           image_path=file_path)


@app.route("/predict-live", methods=["POST"])
def predict_live():
    file = request.files["file"]

    file_path = "static/temp_live.jpg"
    file.save(file_path)

    label, confidence, decision = predict_image(file_path)

    return jsonify({
        "class": label,
        "confidence": confidence,
        "decision": decision
    })


if __name__ == "__main__":
    app.run(debug=True)