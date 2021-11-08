import os
import gc
import io
import time
import base64
import logging

import numpy as np
from PIL import Image

from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS

import detect


app = Flask(__name__)
CORS(app)

u2net = detect.load_model(model_name="u2net")
u2netp = detect.load_model(model_name="u2netp")

logging.basicConfig(level=logging.INFO)


def process_image(net, byte_data: io.BytesIO) -> io.BytesIO:
    start = time.time()
    img = Image.open(byte_data)
    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BILINEAR)  # remove resample
    empty_img = Image.new("RGBA", (img.size), 0)
    new_img = Image.composite(img, empty_img, output.convert("L"))
    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)
    elapsed = time.time() - start
    return buffer, elapsed


@app.route("/", methods=["GET"])
def ping():
    return "U^2-Net!"


@app.route("/remove", methods=["POST"])
def remove():
    start = time.time()
    if 'file' not in request.files:
        return jsonify({'error': 'missing file'}), 400
    if request.files['file'].filename.rsplit('.', 1)[1].lower() not in ["jpg", "png", "jpeg"]:
        return jsonify({'error': 'invalid file format'}), 400
    data = request.files['file'].read()
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400
    buffer, elapsed = process_image(io.BytesIO(data))
    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


@app.route('/upload')
def upload():
    return render_template('upload.html')


def load_image_base64(base64_str):
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")
    imgdata = base64.b64decode(base64_str)
    return imgdata


def process_json(net, json_content):
    if 'image' not in json_content:
        return jsonify({'error': 'missing image file'}), 400
    b64imgSrc = json_content['image']
    if len(b64imgSrc) == 0:
        return jsonify({'error': 'empty image'}), 400
    data = load_image_base64(b64imgSrc)
    buffer, elapsed = process_image(net, io.BytesIO(data))
    b64imgDst = base64.b64encode(buffer.getvalue()).decode()
    return jsonify({
        'elapsed': elapsed,
        'src': b64imgSrc,
        'dst': b64imgDst,
    })


@app.route('/process', methods=['POST'])
def processImage():
    return process_json(u2net, request.json)


@app.route('/process_lite', methods=['POST'])
def processImageLite():
    return process_json(u2netp, request.json)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
