import os
import time
import logging

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from image import load_b64image, extract_image, extract_cheque, image_to_bytes, encode_image_base64

import detect


app = Flask(__name__)
CORS(app)

u2net = detect.load_model(model_name="u2net")
u2netLite = detect.load_model(model_name="u2netp")

logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET"])
def ping():
    return "OK!"


@app.route('/upload')
def uploadImage():
    return render_template('upload.html')


def buildJsonErro(error_msg):
    return jsonify({'error': error_msg})


@app.route('/cheque/extract', methods=['POST'])
def chequeExtract():
    try:
        json_content = request.json

        if 'image' not in json_content:
            return buildJsonErro('missing image file'), 400

        b64imgSrc = json_content['image']
        if len(b64imgSrc) == 0:
            return buildJsonErro('input image is empty'), 400

        debug = json_content.get('debug', False)
        useLite = json_content.get('useLite', False)
        theNet = u2netLite if useLite else u2net

        data = load_b64image(b64imgSrc)

        start = time.time()
        dst_image, edge_image, err = extract_cheque(theNet, data)
        elapsed = time.time() - start

        dst_buffer = image_to_bytes(dst_image)
        b64imgDst = encode_image_base64(dst_buffer)

        response = {
            'elapsed': elapsed,
        }

        if useLite:
            response['useLite'] = useLite

        if err is not None:
            response['error'] = err
        else:
            response['result'] = b64imgDst

        if debug:
            edge_buffer = image_to_bytes(edge_image)
            b64imgEdge = encode_image_base64(edge_buffer)
            response['edge'] = b64imgEdge
            response['result'] = b64imgDst

        return jsonify(response)

    except Exception as e:
        return buildJsonErro(str(e)), 500


@app.route('/image/extract', methods=['POST'])
def imageExtract():
    json_content = request.json
    try:
        if 'image' not in json_content:
            return buildJsonErro('missing image file'), 400

        b64imgSrc = json_content['image']
        if len(b64imgSrc) == 0:
            return buildJsonErro('input image is empty'), 400

        useLite = json_content.get('useLite', False)
        theNet = u2netLite if useLite else u2net

        data = load_b64image(b64imgSrc)

        start = time.time()
        dst_image = extract_image(theNet, data)
        elapsed = time.time() - start

        dst_buffer = image_to_bytes(dst_image)
        b64imgDst = encode_image_base64(dst_buffer)

        response = {
            'elapsed': elapsed,
            'result': b64imgDst,
        }

        if useLite:
            response['useLite'] = useLite

        return jsonify(response)

    except Exception as e:
        return buildJsonErro(str(e)), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
