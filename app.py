from flask import Flask, request, jsonify, send_file
import os
import cv2
import uuid
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = Flask(__name__)

# Set up face analysis
faceapp = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
faceapp.prepare(ctx_id=0, det_size=(640, 640))

# Model path for face swapper
model_path = "inswapper_128.onnx"

# Download model if not present
if not os.path.exists(model_path):
    print("Model not found. Downloading from: https://huggingface.co/akhaliq/inswapper/resolve/main/inswapper_128.onnx")
    import urllib.request
    urllib.request.urlretrieve(
        "https://huggingface.co/akhaliq/inswapper/resolve/main/inswapper_128.onnx",
        model_path
    )

# Load swapper model
try:
    swapper = get_model(model_path, download=False, providers=["CPUExecutionProvider"])
except Exception as e:
    print("Failed to load swapper model:", e)
    exit(1)

@app.route('/swap', methods=['POST'])
def swap_faces():
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'error': 'Missing source or target image'}), 400

    source_image = request.files['source']
    target_image = request.files['target']

    source_path = f"/tmp/{uuid.uuid4().hex}_source.jpg"
    target_path = f"/tmp/{uuid.uuid4().hex}_target.jpg"
    output_path = f"/tmp/{uuid.uuid4().hex}_output.jpg"

    source_image.save(source_path)
    target_image.save(target_path)

    # Load images
    src_img = cv2.imread(source_path)
    tgt_img = cv2.imread(target_path)

    # Detect face in source
    src_faces = faceapp.get(src_img)
    if len(src_faces) == 0:
        return jsonify({'error': 'No face found in source image'}), 400

    src_face = src_faces[0]

    # Detect and swap in target
    tgt_faces = faceapp.get(tgt_img)
    if len(tgt_faces) == 0:
        return jsonify({'error': 'No face found in target image'}), 400

    for tgt_face in tgt_faces:
        tgt_img = swapper.get(tgt_img, tgt_face, src_face, paste_back=True)

    # Save and return result
    cv2.imwrite(output_path, tgt_img)
    return send_file(output_path, mimetype='image/jpeg')

@app.route('/')
def index():
    return 'FaceSwap API is running ✅'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

