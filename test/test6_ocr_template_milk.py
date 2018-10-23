'''OCR TEMPLATE '''

from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from pytesseract import *
import keras
from sklearn.externals import joblib
from skimage.feature import hog

large = cv2.imread('../images/test-digital/test_new.jpg')
rgb = imutils.resize(large, width=1920)  # cv2.pyrDown(large)
# small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image1", small)
# cv2.waitKey(1000)

# kernel_filter = np.array([[-0.1, -1, -0.1], [-1,5, -1], [-0.1, -1, -0.1]], np.float32)  # 锐化
# small = cv2.filter2D(small, -1, kernel=kernel_filter)

# cv2.imwrite('small.jpg',small)
# cv2.imshow("Image3", small)
# cv2.waitKey(1000)

hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
threshold_img = np.zeros((hsv.shape[0], hsv.shape[1], 1), dtype=np.uint8)
cv2.inRange(hsv, np.array([150, 140, 140], np.uint8), np.array([180, 254, 254], np.uint8), threshold_img)
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, sqKernel)
# cv2.imshow("Image4", threshold_img)
# cv2.waitKey(1000)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
# grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
#
# _, bw = cv2.threshold(grad, 140, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
# connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# cv2.imshow("Image5", connected)
# cv2.waitKey(1000)

# using RETR_EXTERNAL instead of RETR_CCOMP
trans = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
cnts = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[1]

_, bw = cv2.threshold(trans, 140, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
mask = np.zeros(bw.shape, dtype=np.uint8)

locs = []

for idx in range(len(cnts)):
    x, y, w, h = cv2.boundingRect(cnts[idx])
    mask[y:y + h, x:x + w] = 0
    cv2.drawContours(mask, cnts, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

    if r > 0.5 and w > 50 and h < 50 and h > 20 and y > 280 and y < 500:
        # cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        locs.append((x, y, w, h))

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x: x[0])
output = []
# cv2.imshow('rects', rgb)
# cv2.waitKey(1000)
# loop over the 4 groupings of 4 digits
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # extract the group ROI of 4 digits from the grayscale image,
    # then apply thresholding to segment the digits from the
    # background of the credit card
    group = trans[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # cv2.imshow('rects0', group)
    # cv2.waitKey(1000)
    group = cv2.threshold(group, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # kernel_filter = np.array([[0, -1.1, 0], [-1, 5.5, -1], [0, -1.1, 0]], np.float32)
    # group = cv2.filter2D(group, -1, kernel=kernel_filter)
    kernel = np.ones((2, 2), np.uint8)
    group = cv2.erode(group, kernel, iterations=2)


    # rotate func
    def rotate(image, angle, center=None, scale=1.0):
        # get the size
        (h, w) = image.shape[:2]

        # if no center w h should be used
        if center is None:
            center = (w / 2, h / 2)

        # rotate it
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


    group = rotate(group, 0)
    # group = cv2.threshold(group, 140, 255, cv2.THRESH_BINARY_INV)[1]
    # group = cv2.Laplacian(group, cv2.CV_64F)
    cv2.imshow('rects1', group)
    cv2.waitKey(1000)
    print(pytesseract.image_to_string(group, lang='eng', boxes=False,
                                      config='-c tessedit_char_whitelist=0123456789 -psm 6'))

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]

    predicts = []
    model = keras.models.load_model('../images/test-digital/fine-tuned-mnist-weights.h5')
    # Load the classifier
    clf = joblib.load("digits_cls.pkl")
    # loop over the digit contours
    for c in digitCnts:
        # compute the bounding box of the individual digit, extract
        # the digit, and resize it to have the same fixed size as
        # the reference OCR-A images
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.copyMakeBorder(roi, 5, 5, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # cv2.imshow('rects', roi)
        # cv2.waitKey(1000)
        roi = cv2.resize(roi, (90, 120), interpolation=cv2.INTER_NEAREST)
        roi = cv2.copyMakeBorder(roi, 15, 15, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        kernel = np.ones((1, 1), np.uint8)
        roi = cv2.erode(roi, kernel, iterations=2)
        cv2.imshow('rects', roi)
        cv2.waitKey(1000)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        kernel = np.ones((1, 1), np.uint8)
        roi = cv2.erode(roi, kernel, iterations=2)
        # roi = rotate(roi, 5)
        # roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow('rects', roi)
        cv2.waitKey(1000)
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        print(nbr)
        #native_digital_al(roi)
        predicts.append(np.array(roi).reshape((28 * 28)))

    predicts = np.asarray(predicts) / 255.
    result = model.predict(predicts)
    print('keras: ', np.argmax(result, axis=1))

    # draw the digit classifications around the group
    cv2.rectangle(rgb, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    cv2.putText(rgb, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # update the output digits list
    output.extend(groupOutput)

# cv2.imshow('rects', rgb)
# cv2.waitKey(10000)
