
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
CELEB_FOLDER = "celebrities"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0)
swapper = get_model('inswapper_128.onnx', download=True)
swapper.prepare(ctx_id=0)

@app.route('/faceswap_with_list/<target_name>', methods=['POST'])
def face_swap_with_list(target_name):
    source = request.files.get('source')
    if not source:
        return jsonify({"error": "Missing source image"}), 400

    target_path = os.path.join(CELEB_FOLDER, target_name)
    if not os.path.exists(target_path):
        return jsonify({"error": "Target image not found"}), 404

    src_path = os.path.join(UPLOAD_FOLDER, f"src_{uuid.uuid4().hex}.jpg")
    source.save(src_path)

    img_src = cv2.imread(src_path)
    img_tgt = cv2.imread(target_path)

    faces_src = face_analyzer.get(img_src)
    faces_tgt = face_analyzer.get(img_tgt)

    if not faces_src or not faces_tgt:
        return jsonify({"error": "Face not detected"}), 400

    swapped_img = swapper.get(img_tgt, faces_tgt[0], faces_src[0], paste_back=True)
    result_path = os.path.join(UPLOAD_FOLDER, f"result_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(result_path, swapped_img)

    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
