import os
import requests
from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
import numpy as np

app = Flask(__name__)

# ---------- Function: Download model if not exists ----------
def download_model_if_missing(model_path, url):
    if not os.path.exists(model_path):
        print("Model not found. Downloading from:", url)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        r = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(r.content)
        print("Download complete.")

# ---------- Initialize InsightFace ----------
print("Initializing FaceAnalysis...")
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0)

# ---------- Ensure model is available ----------
model_path = ".insightface/models/inswapper_128.onnx"
model_url = "https://huggingface.co/akhaliq/inswapper/resolve/main/inswapper_128.onnx"
download_model_if_missing(model_path, model_url)

print("Loading inswapper_128 model...")
swapper = get_model(model_path, download=False)
swapper.prepare(ctx_id=0)

# ---------- Flask API ----------
@app.route('/swap', methods=['POST'])
def swap_faces():
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'error': 'Both source and target images are required.'}), 400

    source_img = cv2.imdecode(np.frombuffer(request.files['source'].read(), np.uint8), cv2.IMREAD_COLOR)
    target_img = cv2.imdecode(np.frombuffer(request.files['target'].read(), np.uint8), cv2.IMREAD_COLOR)

    source_faces = face_analyzer.get(source_img)
    target_faces = face_analyzer.get(target_img)

    if len(source_faces) == 0 or len(target_faces) == 0:
        return jsonify({'error': 'Face not detected in one or both images.'}), 400

    # Assume using first face
    result_img = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

    _, buffer = cv2.imencode('.jpg', result_img)
    response = buffer.tobytes()

    return response, 200, {'Content-Type': 'image/jpeg'}

# ---------- Main ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
