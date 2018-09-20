'''The MIT License (MIT)

Copyright (c) 2017 Dhanushka Dangampola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.'''

from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from pytesseract import *
import keras
from keras.models import Model,Sequential


large = cv2.imread('../images/test-account/test5.jpg')
rgb = imutils.resize(large, width=1920)#cv2.pyrDown(large)
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
cv2.inRange(hsv,np.array([150, 140, 140],np.uint8), np.array([180, 254, 254],np.uint8) , threshold_img)
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow("Image4", threshold_img)
cv2.waitKey(1000)

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
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, cnts, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.5 and w > 50 and h < 50 and h > 20 and y > 280 and y < 500:
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        locs.append((x, y, w, h))

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x:x[0])
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
    kernel_filter = np.array([[0, -1.1, 0], [-1, 5, -1], [0, -1.1, 0]], np.float32)
    group = cv2.filter2D(group, -1, kernel=kernel_filter)
    kernel = np.ones((2, 2), np.uint8)
    group = cv2.erode(group, kernel, iterations=3)
    # 定义旋转rotate函数
    def rotate(image, angle, center=None, scale=1.0):
        # 获取图像尺寸
        (h, w) = image.shape[:2]

        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        # 返回旋转后的图像
        return rotated

    group = rotate(group,5)
    # group = cv2.threshold(group, 140, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('rects1', group)
    cv2.waitKey(1000)
    print(pytesseract.image_to_string(group,lang='eng', boxes=False,config='-c tessedit_char_whitelist=0123456789 -psm 6'))

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
    digitCnts = contours.sort_contours(digitCnts,
                                   method="left-to-right")[0]

    predicts = []
    model = keras.models.load_model('../images/test-account/fine-tuned-mnist-weights.h5')
    # loop over the digit contours
    for c in digitCnts:
        # compute the bounding box of the individual digit, extract
        # the digit, and resize it to have the same fixed size as
        # the reference OCR-A images
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (60, 90))
        roi = cv2.copyMakeBorder(roi, 15, 15, 25, 25, cv2.BORDER_CONSTANT, value=[0,0,0])
        kernel = np.ones((2, 2), np.uint8)
        roi = cv2.erode(roi, kernel, iterations=5)
        cv2.imshow('rects', roi)
        cv2.waitKey(1000)
        roi = cv2.resize(roi, (28, 28))
        # roi = rotate(roi, 5)
        # roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow('rects', roi)
        cv2.waitKey(1000)
        predicts.append(np.array(roi).reshape((28*28)))

    predicts = np.asarray(predicts)/255.
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
