import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# === Inisialisasi Flask App ===
app = Flask(__name__)

# === Load Model ===
MODEL_PATH = 'resnet50_clahe_augmented_balanced_model.h5'
model = load_model(MODEL_PATH)

# === Class Names (urutan sesuai training) ===
class_names = [
    "Dermatitis perioral", "Eksim", "Karsinoma", "Pustula", "Tinea facialis",
    "Acne fulminans", "Acne nodules", "Blackhead", "Flek hitam", "Folikulitis",
    "Fungal acne", "Herpes", "Kutil filiform", "Melanoma", "Milia",
    "Normal", "Panu", "Papula", "Psoriasis", "Rosacea", "Whitehead"
]

IMG_SIZE = (224, 224)

# === Fungsi Preprocessing (CLAHE + ResNet Preprocessing) ===
def preprocess_image(image_bytes):
    # Convert bytes ke array OpenCV
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Gambar tidak valid")

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # RGB & ResNet Preprocessing
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    final = final.astype(np.float32)
    final = preprocess_input(final)
    final = np.expand_dims(final, axis=0)  # shape: (1, 224, 224, 3)
    return final

# === Endpoint Prediksi ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        image = preprocess_image(file.read())
        predictions = model.predict(image)
        pred_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        result = {
            'class': class_names[pred_class],
            'confidence': round(confidence, 4)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run Server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
