import cv2 as cv
import math
import numpy as np


class Target:
    def __init__(self):
        cv.namedWindow("Target", 1)

    def run(self, img_path):
        # capture the image from the cam
        img = cv.imread(img_path)
        # convert the image to HSV
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        corners = cv.goodFeaturesToTrack(gray_img,50,0.01,0.5)
        corners = np.int0(corners)

        for corner in corners:
            x,y = corner.ravel()
            cv.circle(img, (x,y), 3, 255 , -1)
            print(x,y)
        # display frames to users
        cv.imshow("Target", img)
        cv.waitKey(0)

if __name__ == "__main__":
    t = Target()
    t.run('f:/test.jpg')