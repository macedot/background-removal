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
u2netLite = detect.load_model(model_name="u2netp")

logging.basicConfig(level=logging.INFO)


def modelPredict(net, img):
    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BILINEAR)  # remove resample
    output = output.convert("L")
    return output


def imageCompose(img_base, img_mask):
    empty_img = Image.new("RGBA", (img_base.size), 0)
    new_img = Image.composite(img_base, empty_img, img_mask)
    new_img = new_img.convert('L')
    return new_img


def process_image(net, byte_data: io.BytesIO, img_format: str = 'JPEG') -> io.BytesIO:
    start = time.time()
    img = Image.open(byte_data)
    img_mask = modelPredict(net, img)
    new_img = imageCompose(img, img_mask)
    buffer = io.BytesIO()
    new_img.save(buffer, img_format)
    buffer.seek(0)
    elapsed = time.time() - start
    return buffer, elapsed


@app.route("/", methods=["GET"])
def ping():
    return "OK!"


@app.route('/upload')
def upload():
    return render_template('upload.html')


def load_image_base64(base64_str):
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")
    imgdata = base64.b64decode(base64_str)
    return io.BytesIO(imgdata)


def encode_image_base64(image_bytes: io.BytesIO) -> str:
    with io.BytesIO() as output_bytes:
        PIL_image = Image.open(image_bytes)
        if PIL_image.mode != 'L':
            PIL_image = PIL_image.convert('L')
        PIL_image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str


def process_json(json_content):
    if 'image' not in json_content:
        return jsonify({'error': 'missing image file'}), 400
    b64imgSrc = json_content['image']
    if len(b64imgSrc) == 0:
        return jsonify({'error': 'empty image'}), 400
    isLite = json_content.get('isLite', False)
    theNet = u2netLite if isLite else u2net
    data = load_image_base64(b64imgSrc)
    buffer, elapsed = process_image(theNet, data)
    b64imgDst = encode_image_base64(buffer)
    return jsonify({
        'elapsed': elapsed,
        'dst': b64imgDst,
        'isLite': isLite,
    })


@app.route('/process', methods=['POST'])
def processImage():
    return process_json(request.json)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
