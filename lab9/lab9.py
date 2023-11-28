import cv2
import numpy as np
from matplotlib import pyplot as plt

imgOrig = cv2.imread('ATU1.jpg')
imgOrig2 = cv2.imread('ATU2.jpg')

# Assign rows, cols
nrows = 4
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

# Empire State Building ORB
emp1 = cv2.imread('EmpireState1.jpg', cv2.COLOR_BGR2GRAY)
emp2 = cv2.imread('EmpireState2.jpg', cv2.COLOR_BGR2GRAY)

emp1ORB = cv2.ORB_create()
emp2ORB = cv2.ORB_create()

kp1 = emp1ORB.detect(emp1,None)
kp2 = emp2ORB.detect(emp2,None)

kp1, des1 = emp1ORB.compute(emp1, kp1)
kp2, des2 = emp2ORB.compute(emp2, kp2)

empImgORB1 = cv2.drawKeypoints(emp1, kp1, None, color=(255, 0, 255), flags=0)
empImgORB2 = cv2.drawKeypoints(emp2, kp2, None, color=(255, 0, 255), flags=0)

# create BFmatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x:x.distance)

# Draw first 10 matches
empORBMatches = cv2.drawMatches(emp1, kp, emp2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# ATU ORB
atu1 = cv2.imread('ATU1.jpg', cv2.COLOR_BGR2GRAY)
atu2 = cv2.imread('ATU2.jpg', cv2.COLOR_BGR2GRAY)

atu1ORB = cv2.ORB_create()
atu2ORB = cv2.ORB_create()

kp3 = atu1ORB.detect(atu1,None)
kp4 = atu2ORB.detect(atu2,None)

kp3, des3 = atu1ORB.compute(atu1, kp3)
kp4, des4 = atu2ORB.compute(atu2, kp4)

atuImgORB1 = cv2.drawKeypoints(atu1, kp3, None, color=(255, 0, 255), flags=0)
atuImgORB2 = cv2.drawKeypoints(atu2, kp4, None, color=(255, 0, 255), flags=0)

# create BFmatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches2 = bf.match(des3, des4)

# Sort them in the order of their distance
matches2 = sorted(matches2, key=lambda x:x.distance)

# Draw first 10 matches
atuORBMatches = cv2.drawMatches(atu1, kp3, atu2, kp4, matches2[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#FLANN Matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches3 = flann.knnMatch(des3, des4, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches3))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches3):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (255, 0, 255),
                     singlePointColor = (255, 0, 255),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)

atuFLANNMatches = cv2.drawMatchesKnn(atu1, kp3, atu2, kp4, matches3, None, **draw_params)


# Contours detection
contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours = imgOrig.copy()
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

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
plt.subplot(nrows, ncols,6),plt.imshow(empImgORB1, cmap = 'gray')
plt.title('Empire State 1'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,7),plt.imshow(empImgORB2, cmap = 'gray')
plt.title('Empire State 2'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,8),plt.imshow(empORBMatches, cmap = 'gray')
plt.title('Empire State Matches'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,9),plt.imshow(atuImgORB1, cmap = 'gray')
plt.title('ATU 1'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,10),plt.imshow(atuImgORB2, cmap = 'gray')
plt.title('ATU 2'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,11),plt.imshow(atuORBMatches, cmap = 'gray')
plt.title('ATU BF Matches'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,12),plt.imshow(atuFLANNMatches, cmap = 'gray')
plt.title('ATU FLANN Matches'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,13),plt.imshow(imgContours, cmap = 'gray')
plt.title('Contours'), plt.xticks([]), plt.yticks([])


plt.show()

cv2.waitKey(0)
cv2.destroyAllwindows()