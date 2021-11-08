import skimage.io
import skimage.transform
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from PIL import Image
import tensorflow as tf
import numpy as np
import base64
import cv2
from hashlib import sha256
from io import BytesIO
from transform import rectify, four_point_transform


model = tf.keras.models.load_model('../4unet-cheque')


def mm2pix(x, d):
    return (x * d) / 25.4


UNET_IMAGE_WIDTH = 256
UNET_IMAGE_HEIGHT = UNET_IMAGE_WIDTH
UNET_IMAGE_SHAPE = (UNET_IMAGE_HEIGHT, UNET_IMAGE_WIDTH)
CHEQUE_WIDTH_MM = 176
CHEQUE_HEIGHT_MM = 76
CHEQUE_IMAGE_DPI = 200
CHEQUE_DIM_PIX = np.array([
    mm2pix(CHEQUE_WIDTH_MM, CHEQUE_IMAGE_DPI),
    mm2pix(CHEQUE_HEIGHT_MM, CHEQUE_IMAGE_DPI)
], dtype=np.int32)


def load_image_base64(base64_str):
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")
    imgdata = base64.b64decode(base64_str)
    img = skimage.io.imread(imgdata, plugin='imageio', as_gray=True)
    h, w = img.shape
    if h > w:
        img = skimage.transform.rotate(img, 90, resize=True)
    return img


def convert_image_to_model(img):
    digest = sha256(img).hexdigest()
    img = skimage.transform.resize(img, UNET_IMAGE_SHAPE, anti_aliasing=False)
    #skimage.io.imsave(f'{digest}.jpg', img)
    img = np.reshape(img, (1,) + UNET_IMAGE_SHAPE + (1,))
    return img


def encode_image_base64(PIL_image) -> str:
    with BytesIO() as output_bytes:
        if PIL_image.mode != 'RGB':
            PIL_image = PIL_image.convert('RGB')
        PIL_image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str


def run_model(img_src):
    img_input = convert_image_to_model(img_src)
    results = model.predict([img_input], 1, verbose=1)
    img = results[0][:, :, 0]
    img_prediction = Image.fromarray(img_as_ubyte(img))
    return img_prediction


def auto_canny(image, sigma=0.33, apertureSize=7):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, apertureSize)
    return edged


def detect_edge(img):
    im_np = np.asarray(img)
    _, threshold = cv2.threshold(im_np, 1, 255, cv2.THRESH_BINARY)
    threshold = auto_canny(threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    target = None
    imgLocal = img.copy()
    maxArea = 0.10 * UNET_IMAGE_WIDTH * UNET_IMAGE_HEIGHT
    for c in contours:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > maxArea:
                maxArea = area
                target = rectify(approx)
                imgLocal = im_np.copy()
                cv2.fillConvexPoly(imgLocal, target, (255, 255, 255), cv2.LINE_AA)
                break
    image = img_as_ubyte(imgLocal)
    return image, target  # , maxArea


def warp_image(img_src, img_edge, target):
    src_shape = img_src.shape
    edge_shape = img_edge.shape
    adj = np.flip([a / b for a, b in zip(src_shape, edge_shape)])
    roi = np.zeros(target.shape, dtype=np.int32)
    for i, elem in enumerate(target):
        roi[i] = [a * b for a, b in zip(elem, adj)]
    img_dst = four_point_transform(img_src, roi, CHEQUE_DIM_PIX)
    return img_dst


def process_image(b64src: str):
    img_src = load_image_base64(b64src)
    img_prediction = run_model(img_src)
    img_edge, target = detect_edge(img_prediction)
    if target is None:
        return None
    img_dst = warp_image(img_src, img_edge, target)
    dst_pil = Image.fromarray(img_as_ubyte(img_dst))
    b64dst = encode_image_base64(dst_pil)
    return b64dst
