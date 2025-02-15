from scipy.spatial import distance as dist
import numpy as np
import cv2

# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com


def order_points_old(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


# https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
def order_points_2(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify(h):
    h = h.reshape((4, 2))
    h = order_points(h)
    hnew = np.zeros((4, 2), dtype=np.int32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]  # tl
    hnew[2] = h[np.argmax(add)]  # bl
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]  # tr
    hnew[3] = h[np.argmax(diff)]  # br
    return hnew

# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com


def four_point_transform(image, pts, finalDims=None):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    calcWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    calcHeight = max(int(heightA), int(heightB))

    if finalDims is None:
        maxWidth, maxHeight = calcWidth, calcHeight
    else:
        maxWidth, maxHeight = finalDims

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    if calcHeight > calcWidth:
        # // 90 graus COUNTER-CLOCK WISE da versão HORIZONTAL
        # const cv::Point2f OUTPUT_QUAD_VERTICAL[4]{
        #   cv::Point2f(0, FINAL_IMG_HEIGHT - 1),
        #   cv::Point2f(0, 0),
        #   cv::Point2f(FINAL_IMG_WIDTH - 1, 0),
        #   cv::Point2f(FINAL_IMG_WIDTH - 1, FINAL_IMG_HEIGHT - 1),
        # };
        dst = np.array([
            [0, maxHeight - 1],
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
        ], dtype=np.float32)
    else:
        # const cv::Point2f OUTPUT_QUAD_HORIZONTAL[4]{
        #   cv::Point2f(0, 0),
        #   cv::Point2f(FINAL_IMG_WIDTH - 1, 0),
        #   cv::Point2f(FINAL_IMG_WIDTH - 1, FINAL_IMG_HEIGHT - 1),
        #   cv::Point2f(0, FINAL_IMG_HEIGHT - 1),
        # };
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ], dtype=np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
