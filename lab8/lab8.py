import cv2
import numpy as np
from matplotlib import pyplot as plt

print("Hello World")

imgOrig = cv2.imread('ATU.jpg')

nrows = 2
ncols = 2


imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
imgOut = cv2.GaussianBlur(imgGray,(5, 5),0)
blur = cv2.blur(imgGray,(13, 13),0)

plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgOut, cmap = 'gray')
plt.title('GuassianBlur'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(blur, cmap = 'gray')
plt.title('blur'), plt.xticks([]), plt.yticks([])
plt.show()




cv2.waitKey(0)
cv2.destroyallwindows()


