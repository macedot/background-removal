import io
import base64
import logging
import numpy as np
import cv2 as cv2
from PIL import Image
from PIL.ImageFilter import (UnsharpMask)
from transform import rectify, four_point_transform

import detect


def mm2pix(x, d):
    return (x * d) / 25.4


CHEQUE_WIDTH_MM = 176
CHEQUE_HEIGHT_MM = 76
CHEQUE_IMAGE_DPI = 200
CHEQUE_DIM_PIX = np.array([
    mm2pix(CHEQUE_WIDTH_MM, CHEQUE_IMAGE_DPI),
    mm2pix(CHEQUE_HEIGHT_MM, CHEQUE_IMAGE_DPI),
], dtype=np.int32)

CHEQUE_DIM_PIX_VERT = np.array([
    mm2pix(CHEQUE_HEIGHT_MM, CHEQUE_IMAGE_DPI),
    mm2pix(CHEQUE_WIDTH_MM, CHEQUE_IMAGE_DPI),
], dtype=np.int32)


def load_b64image(base64_str):
    if isinstance(base64_str, bytes):
        base64_str = base64_str.decode("utf-8")
    imgdata = base64.b64decode(base64_str)
    return io.BytesIO(imgdata)


def encode_image_base64(PIL_image, img_mode: str = 'RGBA', img_format: str = 'PNG') -> str:
    with io.BytesIO() as output_bytes:
        if PIL_image.mode != img_mode:
            PIL_image = PIL_image.convert(img_mode)
        PIL_image.save(output_bytes, img_format)
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str


def modelPredict(net, img, doUnsharp: bool = False):
    if doUnsharp:
        #img = img.filter(UnsharpMask(radius=4.5, percent=200, threshold=0))
        #img = img.filter(UnsharpMask(radius=3, percent=200, threshold=5))
        #img = img.filter(UnsharpMask(radius=1.5, percent=75, threshold=10))
        img = img.filter(UnsharpMask())
    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BILINEAR)
    output = output.convert("L")
    return output


def imageCompose(img_base, img_mask, img_type: str = 'RGBA'):
    empty_img = Image.new(img_type, (img_base.size), 0)
    new_img = Image.composite(img_base.convert(img_type), empty_img, img_mask)
    return new_img


def auto_canny(image, sigma=0.33, apertureSize=7):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, apertureSize)
    return edged


def detect_edge(img):
    im_np = np.asarray(img)
    #_, threshold = cv2.threshold(im_np, 1, 255, cv2.THRESH_BINARY)
    _, threshold = cv2.threshold(im_np, 1, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold = auto_canny(threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    imgLocal = im_np.copy()
    target = None
    maxArea = 0.10 * im_np.shape[0] * im_np.shape[1]
    for c in contours:
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.01 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            approx = rectify(approx)
            if area > maxArea:
                target = approx
                maxArea = area
                cv2.fillConvexPoly(imgLocal, target, (255, 255, 255), cv2.LINE_AA)
                break
            else:
                cv2.fillConvexPoly(imgLocal, approx, (255, 255, 255), cv2.LINE_AA)
    return imgLocal, target


def remove_shadown(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm


def image_auto_adjustment(img):
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0
    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def warp_image(img_src, img_edge, target):
    src_shape = img_src.shape
    edge_shape = img_edge.shape
    adj = np.flip([a / b for a, b in zip(src_shape, edge_shape)])
    roi = np.zeros(target.shape, dtype=np.int32)
    for i, elem in enumerate(target):
        roi[i] = [a * b for a, b in zip(elem, adj)]
    img_dst = four_point_transform(img_src, roi, CHEQUE_DIM_PIX)
    img_dst = image_auto_adjustment(img_dst)
    return Image.fromarray(img_dst)


def extract_image(net, byte_data: io.BytesIO, doUnsharp: bool = False):
    img = Image.open(byte_data)
    img_mask = modelPredict(net, img, doUnsharp)
    img_final = imageCompose(img, img_mask)
    return img_final


def extract_cheque(net, byte_data: io.BytesIO, doUnsharp: bool = False):
    msg = None
    img = Image.open(byte_data)
    img_mask = modelPredict(net, img, doUnsharp)
    img_edge, target = detect_edge(img_mask)
    if target is None:
        img_dst = img_mask
        msg = 'No region found'
    else:
        img_dst = warp_image(np.asarray(img), np.asarray(img_edge), target)

    return img_dst, Image.fromarray(img_edge), msg
