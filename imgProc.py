import cv2
import numpy as np
#import subprocess

img = cv2.imread('sud.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(imgray, (11, 11), 0)
th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,2)
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
erosion = cv2.erode(th, kernel, iterations = 4)

contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
maxA = cv2.contourArea(contours[0], True)
max_i = 0
for i in range(1, len(contours)) :
	area = cv2.contourArea(contours[i], True)
	if area > maxA :
		maxA = area
		max_i = i

mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask, contours, max_i, 255, -1)
pixelpoints = np.nonzero(mask)

X = pixelpoints[1]
Y = pixelpoints[0]

SUM = X + Y
DIFF = X - Y

a1 = np.argmax(SUM)
a2 = np.argmin(SUM)
a3 = np.argmax(DIFF)
a4 = np.argmin(DIFF)

pts1 = np.float32([[X[a2], Y[a2]], [X[a3], Y[a3]], [X[a1], Y[a1]], [X[a4], Y[a4]]])
pts2 = np.float32([[0,0],[303,0],[303,303],[0,303]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(imgray,M,(303,303))

# cv2.drawContours(img, contours, max_i, (0,255,0), 3)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600, 600)
cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()