import cv2 as cv
import math
import numpy as np


class Target:
    def __init__(self):
        self.capture = cv.VideoCapture(0)
        cv.namedWindow("Target", 1)
        cv.namedWindow("Threshold1", 1)
        cv.namedWindow("Threshold2", 1)
        cv.namedWindow("hsv", 1)

    def run(self):
        # initiate font
        font = cv.FONT_HERSHEY_SIMPLEX

        frame_height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))

        # instantiate images
        # hsv_img=cv.CreateImage(cv.GetSize(cv.QueryFrame(self.capture)),8,3)
        hsv_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        threshold_img1 = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        threshold_img1a = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        threshold_img2 = np.zeros((frame_height, frame_width, 1), dtype=np.uint8)
        i = 0

        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv.VideoWriter("angle_tracking.avi", fourcc, 30, (frame_height, frame_width), 1)

        while True:
            # capture the image from the cam
            ret, img = self.capture.read()

            # convert the image to HSV
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            # cv.imshow('ddd1',hsv_img)
            # cv.waitKey(0)
            # threshold the image to isolate two colors
            cv.inRange(hsv_img, np.array([165,145,100],np.uint8), np.array([180,210,160],np.uint8), threshold_img1)  # red
            # cv.imshow('ddd2',threshold_img1)
            # cv.waitKey(0)
            cv.inRange(hsv_img, np.array([0, 145, 100],np.uint8), np.array([190, 255, 255],np.uint8), threshold_img1a)  # red again
            cv.add(threshold_img1, threshold_img1a, threshold_img1)  # this is combining the two limits for red
            cv.inRange(hsv_img,np.array([110,50,50],np.uint8), np.array([130,255,255],np.uint8) , threshold_img2)  # blue

            # cv.imshow('ddd3',threshold_img2)
            # cv.waitKey(0)

            # determine the moments of the two objects
            # threshold_img1=cv.getMat(threshold_img1)
            # threshold_img2=cv.getMat(threshold_img2)
            moments1 = cv.moments(threshold_img1)
            moments2 = cv.moments(threshold_img2)
            area1 = moments1['m00']
            area2 = moments2['m00']

            # initialize x and y
            x1, y1, x2, y2 = (1, 2, 3, 4)
            coord_list = [x1, y1, x2, y2]
            for x in coord_list:
                x = 0

            # there can be noise in the video so ignore objects with small areas
            if (area1 > 100000):
                # x and y coordinates of the center of the object is found by dividing the 1,0 and 0,1 moments by the area
                x1 = int(moments1['m10'] / area1)
                y1 = int(moments1['m01'] / area1)

                # draw circle
                cv.circle(img, (x1, y1), 2, (0, 255, 0), 20)

                # write x and y position
                cv.putText(img, str(x1) + "," + str(y1), (x1, y1 + 20), font, 1, (255, 255, 255))  # Draw the text

            if (area2 > 100000):
                # x and y coordinates of the center of the object is found by dividing the 1,0 and 0,1 moments by the area
                x2 = int(moments2['m10'] / area2)
                y2 = int(moments2['m01'] / area2)

                # draw circle
                cv.circle(img, (x2, y2), 2, (0, 255, 0), 20)

                cv.putText(img, str(x2) + "," + str(y2), (x2, y2 + 20), font, 1, (255, 255, 255))  # Draw the text
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv.LINE_AA)
                # draw line and angle
                cv.line(img, (x1, y1), (frame_width, y1), (100, 100, 100), 2, cv.LINE_AA)

            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            angle = int(math.atan((y1 - y2) / (x2 - x1 + 1e-10)) * 180 / math.pi)
            cv.putText(img, str(angle), (int(x1 + 50.), int(int(y2 + y1) / 2.)), font, 2, (255, 255, 255))

            # cv.writeFrame(writer,img)


            # display frames to users
            cv.imshow("Target", img)
            cv.imshow("Threshold1",threshold_img1)
            cv.imshow("Threshold2",threshold_img2)
            cv.imshow("hsv",hsv_img)
            # Listen for ESC or ENTER key
            c = cv.waitKey(7) % 0x100
            if c == 27 or c == 10:
                break
        cv.destroyAllWindows()


if __name__ == "__main__":
    t = Target()
    t.run()