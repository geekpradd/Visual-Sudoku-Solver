import cv2
import numpy as np

# def fillCol(img, i, j, col):
# 	if i < 0 or i >= img.shape[0] or j < 0 or j >= img.shape[1] or img[i][j] == col or img[i][j] == 0:
# 		return img, 0

# 	img[i][j] = col
# 	img, num1 = fillCol(img, i+1, j, col)
# 	img, num2 = fillCol(img, i-1, j, col)
# 	img, num3 = fillCol(img, i, j+1, col)
# 	img, num4 = fillCol(img, i, j-1, col)
# 	return img, 1+num1+num2+num3+num4

# for i in range(0, 10) :
# 	img = cv2.imread('digits/p'+str(i)+'.jpg', 0)
# 	eh = cv2.equalizeHist(img)
# 	th = np.sum(eh)/(eh.size*4)
# 	print(th)

# 	ret, img2 = cv2.threshold(eh, th, 255, cv2.THRESH_BINARY_INV)
# 	cv2.imshow('image', img2)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
# 	ar = 0
# 	y_m = 0
# 	x_m = 0
# 	for y in range(img2.shape[0]):
# 		for x in range(img2.shape[1]):
# 			if img2[y][x] == 255:
# 				img2, num = fillCol(img2, y, x, 120)
# 				if num > ar:
# 					ar = num
# 					y_m = y
# 					x_m = x

# 	print(ar*100.0/img2.size)

# # cv2.imshow('image', img2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

img = cv2.imread('sud3.jpg', 0)
eh = cv2.equalizeHist(img)
th = np.sum(eh)/(eh.size*4)
ret, img2 = cv2.threshold(eh, th, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)
cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()