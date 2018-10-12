# import the necessary packages
from PIL import Image
import numpy as np
import cv2
import zbar
import yaml
import glob
import os
from matplotlib import pyplot as plt
import math
import logging
import zbarlight
from PIL import Image

logger = logging.getLogger(__name__)
if not logger.handlers: logging.basicConfig(level=logging.INFO)
DEBUG = (logging.getLevelName(logger.getEffectiveLevel()) == 'DEBUG')


def show(img, code=cv2.COLOR_BGR2RGB):
    cv_rgb = cv2.cvtColor(img, code)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(cv_rgb)
    fig.show()


# l2 distance
def cv_distance(P, Q):
    return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]), 2)))


def check(a, b, gray_image):
    s1_ab = ()
    s2_ab = ()
    s1 = np.iinfo('i').max
    s2 = s1
    for ai in a:
        for bi in b:
            d = cv_distance(ai, bi)
            if d < s2:
                if d < s1:
                    s1_ab, s2_ab = (ai, bi), s1_ab
                    s1, s2 = d, s1
                else:
                    s2_ab = (ai, bi)
                    s2 = d
    a1, a2 = s1_ab[0], s2_ab[0]
    b1, b2 = s1_ab[1], s2_ab[1]
    # fix the line according to the QR box rules
    a1 = (a1[0] + (a2[0] - a1[0]) * 1 / 14, a1[1] + (a2[1] - a1[1]) * 1 / 14)
    b1 = (b1[0] + (b2[0] - b1[0]) * 1 / 14, b1[1] + (b2[1] - b1[1]) * 1 / 14)
    a2 = (a2[0] + (a1[0] - a2[0]) * 1 / 14, a2[1] + (a1[1] - a2[1]) * 1 / 14)
    b2 = (b2[0] + (b1[0] - b2[0]) * 1 / 14, b2[1] + (b1[1] - b2[1]) * 1 / 14)
    pix_vector1 = createLineIterator(a1, b1, gray_image)[::, 2]
    pix_vector2 = createLineIterator(a2, b2, gray_image)[::, 2]
    # cv2.line(gray_image, a1, b1, (0,0,255), 3)
    # cv2.imshow("ttt1", gray_image)
    # cv2.waitKey(0)
    # cv2.line(gray_image, a2, b2, (0,0,255), 3)
    # cv2.imshow("ttt1", gray_image)
    # cv2.waitKey(0)
    return isTimingPattern(pix_vector1) or isTimingPattern(pix_vector2)


# Judging if the line is a timing pattern, this function can be really different for each coder
def isTimingPattern(line):
    # get rid of the pixels at the top and end
    # while line[0] != 0:
    #     line = line[1:]
    # while line[-1] != 0:
    #     line = line[:-1]
    # while line[0] == 0:
    #     line = line[1:]
    # while line[-1] == 0:
    #     line = line[:-1]

    c = []
    count = 1
    l = line[0]
    for p in line[1:]:
        if p == l:
            count = count + 1
        else:
            c.append(count)
            count = 1
        l = p
    c.append(count)
    if len(c) < 5:
        return False
    threshold = 200
    # reduce mean
    return np.var(c) < threshold


"""
Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

Parameters:
    -P1: a numpy array that consists of the coordinate of the first point (x,y)
    -P2: a numpy array that consists of the coordinate of the second point (x,y)
    -img: the image being processed

Returns:
    -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
"""


def createLineIterator(P1, P2, img):
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


def count_children(hierarchy, parent, inner=False):
    if parent == -1:
        return 0
    elif not inner:
        return count_children(hierarchy, hierarchy[parent][2], True)
    return 1 + count_children(hierarchy, hierarchy[parent][0], True) + count_children(hierarchy, hierarchy[parent][2],
                                                                                      True)


def has_square_parent(hierarchy, squares, parent):
    if hierarchy[parent][3] == -1:
        return False
    if hierarchy[parent][3] in squares:
        return True
    return has_square_parent(hierarchy, squares, hierarchy[parent][3])


def ocr_qrcode_zbar(image):
    pil = image.convert('L')
    width, height = pil.size
    raw = pil.tobytes()
    # wrap image data
    zarimage = zbar.Image(width, height, 'Y800', raw)
    # create a reader
    scanner = zbar.ImageScanner()
    # configure the reader
    scanner.parse_config('enable')
    # to zar bar image
    scanner.scan(zarimage)

    data = ''
    for symbol in zarimage:
        data += symbol.data
    # if data:
    #     logger.debug(u'QRcode :%s, content: %s' % (filename, data))
    # else:
    #     logger.error(u'Zbar recognize error:%s' % (filename))
    #     img.save('%s-zbar.jpg' % filename)
    return data


def ocr_qrcode_zbarlight(image):
    # zbarlight formatted
    data = zbarlight.scan_codes('qrcode', image)

    # if data:
    #     logger.debug(u'QRcode:%s, content: %s' % (filename, data))
    # else:
    #     logger.error(u'zbarlight recognize error:%s' % (filename))
    #     img.save('%s-zbar.jpg' % filename)
    return data

# do not used
def get_angle(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    return math.degrees(math.atan2(y_diff, x_diff))

# do not used
def get_midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

# do not used
def get_center(c):
    m = cv2.moments(c)
    return [int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])]


# do not used
def get_farthest_points(contour, center):
    distances = []
    distances_to_points = {}
    for point in contour:
        point = point[0]
        d = math.hypot(point[0] - center[0], point[1] - center[1])
        distances.append(d)
        distances_to_points[d] = point
    distances = sorted(distances)
    return [distances_to_points[distances[-1]], distances_to_points[distances[-2]]]


# do not used
def extend(a, b, length, int_represent=False):
    length_ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    if length_ab * length <= 0:
        return b
    result = [b[0] + (b[0] - a[0]) / length_ab * length, b[1] + (b[1] - a[1]) / length_ab * length]
    if int_represent:
        return [int(result[0]), int(result[1])]
    else:
        return result


# do not used
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        return [-1, -1]

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return [int(x), int(y)]


def if_is_qrcode_square(squares):
    qrcodes = []
    minrect = []
    minpoint = []
    # Determine if squares are QR codes
    index = 0
    for arr in squares:
        # arr = sorted(arr, key=lambda bnb: bnb[0][0] or bnb[0][1], reverse=False)
        # calc the center of the contours
        center = get_center(arr)
        holder = np.zeros((4, 2), dtype=np.int32)
        # add the margin , because of the int and float reduce. Broaden the sphere of the squares
        # chose margin = 3
        for point in arr.reshape(4, 2):
            if point[0] - center[0] < 0 and point[1] - center[1] < 0:
                point[0] += -5
                point[1] += -5
                holder[0] = point
            elif point[0] - center[0] > 0 and point[1] - center[1] < 0:
                point[0] += 5
                point[1] += -5
                holder[1] = point
            elif point[0] - center[0] > 0 and point[1] - center[1] > 0:
                point[0] += 5
                point[1] += 5
                holder[2] = point
            elif point[0] - center[0] < 0 and point[1] - center[1] > 0:
                point[0] += -5
                point[1] += 5
                holder[3] = point
        squares[index] = holder
        index += 1

    # sort it by the top left contour dx
    squares = sorted(squares, key=lambda bnb: bnb[0][0], reverse=False)
    for index, arr in enumerate(squares):
        if arr[0][1] < 150 or arr[0][1] > 750:
            squares[index] = None

    def is_none(x):
        return x is not None

    squares = filter(is_none, squares)
    # for python3
    import sys
    if sys.version_info > (3, 0):
        squares = list(squares)

    for index in range(len(squares)):
        if index == len(squares) - 2:
            break
        # get the nearest 3 squares and get the center of each
        square = squares[index]
        square_n1 = squares[index + 1]
        square_n2 = squares[index + 2]
        try:
            center = get_center(square)
            center1 = get_center(square_n1)
            center2 = get_center(square_n2)
        except:
            continue

        # specially for the baoding farm cattles camera image
        if cv_distance(center, center1) < 80 \
                and cv_distance(center1, center2) < 110 \
                and cv_distance(center, center2) < 110:
            qrcodes.append([square, square_n1, square_n2])

            if square[0][1] <= square_n1[0][1]:
                top_left_point = square[0]
                bottom_left_point = square_n1[3]
            else:
                top_left_point = square_n1[0]
                bottom_left_point = square[3]
            top_right_point = square_n2[1]
            # calculate the bottom_right_point using line k and dx dy
            distance_same = cv_distance(top_left_point, bottom_left_point)
            if top_left_point[0] == bottom_left_point[0]:
                brp_x = top_right_point[0]
                brp_y = top_right_point[1] + bottom_left_point[1] - top_left_point[1]
            else:
                k = float(top_left_point[1] - bottom_left_point[1]) / float(top_left_point[0] - bottom_left_point[0])
                patch_x = distance_same / math.sqrt(k ** 2 + 1.)
                if top_left_point[0] - bottom_left_point[0] > 0:
                    brp_x = int(top_right_point[0] - patch_x)
                else:
                    brp_x = int(top_right_point[0] + patch_x)

                brp_y = int(top_right_point[1] + k * brp_x - k * top_right_point[0])

            bottom_right_point = np.array((brp_x, brp_y), dtype=np.int32)

            # format the qrcode area
            temp_points = np.array([], dtype=np.int32).reshape(0, 2)
            temp_points = np.append(temp_points, top_left_point.reshape(1, 2), axis=0)
            temp_points = np.append(temp_points, top_right_point.reshape(1, 2), axis=0)
            temp_points = np.append(temp_points, bottom_right_point.reshape(1, 2), axis=0)
            temp_points = np.append(temp_points, bottom_left_point.reshape(1, 2), axis=0)

            xmin, ymin = np.min(temp_points, axis=0)
            xmax, ymax = np.max(temp_points, axis=0)

            # using minrect and minpoint to store the points and ex value
            minrect.append(temp_points)
            minpoint.append([xmin, xmax, ymin, ymax])
            index += 3
        else:
            index += 1
    return minrect, minpoint


def get_clearly_result(points, image):
    # h, w, ch = image.shape
    # img2 = np.zeros([h, w, ch], image.dtype)
    # image = cv2.addWeighted(image, 0.6, img2, 0.1, 80)
    OriginalImage = image
    Point1 = points[0]
    Point2 = points[1]
    Point3 = points[2]
    src = np.float32([Point1, Point2, Point3])
    # let's make the qr image bigger
    dest_pointTop = [0, 0]
    dest_pointRight = [80, 0]
    dest_pointBottom = [0, 80]
    # transform it
    destination = np.float32(
        [dest_pointTop, dest_pointRight, dest_pointBottom])
    affineTrans = cv2.getAffineTransform(src, destination)
    TransformImage = cv2.warpAffine(
        OriginalImage, affineTrans, (80, 80))
    # show the transformed result
    cv2.imshow("ttt1", TransformImage)
    cv2.waitKey(0)
    # cv2.imshow("ttt2", TransformImage)
    # cv2.waitKey(0)
    print('Original: ', ocr_qrcode_zbar(Image.fromarray(image)))
    print('Original: ', ocr_qrcode_zbarlight(Image.fromarray(image)))
    print('Transformed: ', ocr_qrcode_zbar(Image.fromarray(TransformImage)))
    print('Transformed: ', ocr_qrcode_zbarlight(Image.fromarray(TransformImage)))


def detect(image):
    original_image = image.copy()
    h, w, ch = image.shape
    img2 = np.zeros([h, w, ch], image.dtype)
    # deal with the images make it lighter and sharpened
    image = cv2.addWeighted(image, 0.9, img2, 0.4, 50)
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # keep the edge and rm the noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # blur and threshold the image GaussianBlur
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    # Using Canny function to detect the edge
    edges = cv2.Canny(blurred, 30, 200)
    # find the contours in the thresholded image
    img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # init the holders
    squares = []
    found = []
    i = 0
    SQUARE_TOLERANCE = 0.1
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        # Find all quadrilateral contours
        if len(approx) == 4:
            # Determine if quadrilateral is a square to within SQUARE_TOLERANCE
            if area > 25 \
                    and 1 - SQUARE_TOLERANCE < math.fabs((peri / 4) ** 2) / area < 1 + SQUARE_TOLERANCE \
                    and count_children(hierarchy[0], i) >= 2 \
                    and has_square_parent(hierarchy[0], found, i) is False:
                squares.append(approx)
                found.append(i)
        i += 1
    # jugde if the square is the qrcode edge square and get the qrcode area coordinates
    minrect, minpoint = if_is_qrcode_square(squares)
    # give the margin to avoid the mutate in image
    for index in range(len(minpoint)):
        # testimg is actually the qrcode area
        testimg = original_image[minpoint[index][2]:minpoint[index][3], minpoint[index][0]:minpoint[index][1]]
        point_trans = []
        points = minrect[index]
        xmin = minpoint[index][0]
        ymin = minpoint[index][2]
        # for transforming the image, the left coord should be (0,0)
        point_trans.append([points[0][0] - xmin, points[0][1] - ymin])
        point_trans.append([points[1][0] - xmin, points[1][1] - ymin])
        point_trans.append([points[3][0] - xmin, points[3][1] - ymin])
        # transform and predict result
        get_clearly_result(point_trans, testimg)

    # copy the image for visualization convenient
    draw_img = image.copy()
    boxes = []
    # Get the boxes
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)
        box = map(tuple, box)
        boxes.append(box)

    cv2.imshow('result', draw_img)
    cv2.waitKey(0)


'''MAIN ENTRY'''
paths = ['../images/test-qrcode/0.jpg']

for i in range(len(paths)):
    # load the image
    obj = cv2.imread(paths[i])
    # detect the barcode in the image
    imrg = detect(obj)
    # print(ocr_qrcode_zbarlight(Image.fromarray(imrg)))
    # get the predict result using zbar and zbarlight
    print(ocr_qrcode_zbarlight(Image.open(paths[i])))
    print(ocr_qrcode_zbar(Image.open(paths[i])))

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
