import cv2
import numpy as np
from matplotlib import pyplot as plt

print("Hello World")

imgOrig = cv2.imread('spidey.jpg')

nrows = 2
ncols = 5


imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
imgOut = cv2.GaussianBlur(imgGray,(5, 5),0)
blur = cv2.blur(imgGray,(13, 13),0)

sobelHorizontal = cv2.Sobel(imgGray,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(imgGray,cv2.CV_64F,0,1,ksize=5) # y dir
sobelSum = cv2.addWeighted(sobelVertical, 0.5, sobelHorizontal, 0.5, 0)

canny = cv2.Canny(imgGray,100,250)

threshold = 300
for i in range(0, sobelSum.shape[0]):
    for j in range(0, sobelSum.shape[1]):
        if sobelSum[i,j] > threshold/2:
            sobelSum[i,j] = threshold
        else:
            sobelSum[i,j] = 0

plt.subplot(nrows, ncols,1),plt.imshow(imgOrig, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Fixed Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgGray, cmap = 'gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imgOut, cmap = 'gray')
plt.title('GuassianBlur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(blur, cmap = 'gray')
plt.title('blur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,6),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('sobel Horizontal'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,7),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('sobel Vertical'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,8),plt.imshow(sobelSum, cmap = 'gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,9),plt.imshow(canny, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.show()



cv2.waitKey(0)
cv2.destroyallwindows()


