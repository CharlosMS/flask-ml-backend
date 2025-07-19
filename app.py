from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

# === 1. Load Model ===
model = load_model("resnet50_clahe_augmented_balanced_model.h5")

# === 2. Daftar Kelas (Sesuai Urutan Training) ===
class_names = [
    "Dermatitis perioral", "Eksim", "Karsinoma", "Pustula", "Tinea facialis",
    "acne fulminans", "acne nodules", "blackhead", "flek hitam", "folikulitis",
    "fungal acne", "herpes", "kutil filiform", "melanoma", "milia",
    "panu", "papula", "psoriasis", "rosacea", "whitehead"
]

# === 3. Fungsi Preprocessing (contoh: resize + CLAHE) ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Gambar tidak bisa dibaca oleh OpenCV")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # pastikan ukuran sesuai model

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    final_img = final_img.astype("float32") / 255.0
    final_img = np.expand_dims(final_img, axis=0)

    return final_img

# === 4. Route Prediksi ===
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400

    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    try:
        img = preprocess_image(file_path)
        prediction = model.predict(img)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# === 5. Run App ===
if __name__ == "__main__":
    app.run(debug=True)
