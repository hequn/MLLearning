import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = '../cow_back1.jpg'
imgname2 = '../cow_back2.jpg'

surf = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

img1 = cv2.imread(imgname1)
img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_CUBIC)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kp1, des1 = surf.detectAndCompute(img1, None)

img2 = cv2.imread(imgname2)
img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_CUBIC)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = surf.detectAndCompute(img2, None)

hmerge = np.hstack((gray1, gray2))
cv2.imshow("gray", hmerge)
cv2.waitKey(1)

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))

hmerge = np.hstack((img3, img4))
cv2.imshow("point", hmerge)
cv2.waitKey(1)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

print('The matched points: ', len(matches))
print('The good matched points: ', len(good))

img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow("SURF", img5)
cv2.waitKey(30000)
cv2.destroyAllWindows()
