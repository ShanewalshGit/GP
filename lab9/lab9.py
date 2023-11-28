import cv2
import numpy as np
from matplotlib import pyplot as plt

imgOrig = cv2.imread('ATU1.jpg')

# Assign rows, cols
nrows = 2
ncols = 5

# convert to grayscale, gaussian blur, and blur
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Harris corner detection
blockSize = 2
apertureSize = 3
k = 0.04

dst = cv2.cornerHarris(imgGray, blockSize, apertureSize, k)
imgHarris = imgOrig.copy()

threshold = 0.01 #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
                cv2.circle(imgHarris,(j,i),2,(255, 0, 255),-1)

# Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(imgGray,25,0.01,10)
corners = np.int0(corners)

imgShiTomasi = imgOrig.copy()

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(255, 0, 255),-1)


# ORB corner detection

orb = cv2.ORB_create()
kp = orb.detect(imgGray,None)
kp, des = orb.compute(imgGray, kp)

# draw only keypoints location,not size and orientation
imgORB = cv2.drawKeypoints(imgGray, kp, None, color=(255, 0, 255), flags=0)

spidey1 = cv2.imread('spidey.jpg', cv2.COLOR_BGR2GRAY)
spidey2 = cv2.imread('spidey2.jpg', cv2.COLOR_BGR2GRAY)

Spidey1ORB = cv2.ORB_create()
Spidey2ORB = cv2.ORB_create()

kp1 = Spidey1ORB.detect(spidey1,None)
kp2 = Spidey2ORB.detect(spidey2,None)

kp1, des1 = Spidey1ORB.compute(spidey1, kp1)
kp2, des2 = Spidey2ORB.compute(spidey2, kp2)

SpideyImgORB1 = cv2.drawKeypoints(spidey1, kp1, None, color=(255, 0, 255), flags=0)
SpideyImgORB2 = cv2.drawKeypoints(spidey2, kp2, None, color=(255, 0, 255), flags=0)


# plot images into row and columns with titles
plt.subplot(nrows, ncols,1),plt.imshow(imgOrig, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgHarris, cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imgShiTomasi, cmap = 'gray')
plt.title('Shi-Tomasi'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(imgORB, cmap = 'gray')
plt.title('ORB'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,6),plt.imshow(SpideyImgORB1, cmap = 'gray')
plt.title('Spidey1ORB'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,7),plt.imshow(SpideyImgORB2, cmap = 'gray')
plt.title('Spidey2ORB'), plt.xticks([]), plt.yticks([])


plt.show()

cv2.waitKey(0)
cv2.destroyallwindows()