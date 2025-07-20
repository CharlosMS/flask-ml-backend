import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# === Inisialisasi Flask App ===
app = Flask(__name__)

# === Load Model ===
MODEL_PATH = 'resnet50_clahe_augmented_balanced_model_tanpaEksim.h5'
model = load_model(MODEL_PATH)

# === Class Names (sesuai urutan label saat training) ===
class_names = [
    "Dermatitis perioral", "Karsinoma", "Pustula", "Tinea facialis",
    "acne fulminans", "acne nodules", "blackhead", "flek hitam",
    "folikulitis", "fungal acne", "herpes", "kutil filiform",
    "melanoma", "milia", "normal", "panu", "papula",
    "psoriasis", "rosacea", "whitehead"
]

IMG_SIZE = (224, 224)

# === Fungsi Preprocessing (dengan CLAHE) ===
def preprocess_image(image_bytes):
    try:
        # Convert bytes ke array OpenCV
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Gambar tidak valid atau format tidak didukung")

        # Resize
        img = cv2.resize(img, IMG_SIZE)

        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # RGB & preprocess ResNet
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        final = final.astype(np.float32)
        final = preprocess_input(final)
        final = np.expand_dims(final, axis=0)  # shape: (1, 224, 224, 3)
        return final
    except Exception as e:
        raise ValueError(f"Error dalam preprocessing gambar: {str(e)}")

# === Endpoint Prediksi ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400

    file = request.files['image']
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Format file tidak didukung. Harus JPG/JPEG/PNG'}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'File gambar kosong'}), 400

        image = preprocess_image(image_bytes)
        predictions = model.predict(image)
        
        # Get top 3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_classes = [class_names[i] for i in top3_indices]
        top3_confidences = [float(predictions[0][i]) for i in top3_indices]
        
        result = {
            'top_prediction': {
                'class': class_names[np.argmax(predictions[0])],
                'confidence': round(float(np.max(predictions[0])), 4)
            },
            'top3_predictions': [
                {'class': cls, 'confidence': round(conf, 4)}
                for cls, conf in zip(top3_classes, top3_confidences)
            ],
            'all_predictions': {
                cls: round(float(conf), 4)
                for cls, conf in zip(class_names, predictions[0])
            }
        }
        return jsonify(result)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

# === Health Check Endpoint ===
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})

# === Main ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)