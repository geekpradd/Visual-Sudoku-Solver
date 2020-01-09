import cv2 as cv
import numpy as np 
img_i = cv.imread("sud.jpg") 
img = cv.resize(img_i, (400, 400), interpolation=cv.INTER_AREA)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


initial = np.float32([[31,10],[373,15],[14,377],[394,381]])
final = np.float32([[0,0],[1000,0],[0,1000],[1000,1000]])

M = cv.getPerspectiveTransform(initial, final)

img_f = cv.warpPerspective(img, M, (1000, 1000))

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', 1000, 1000)
cv.imshow('image', img_f)
cv.waitKey(0)
cv.destroyAllWindows()self.weights)