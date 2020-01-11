import cv2
import numpy as np
#import subprocess

neurons = []
biases = []
weights = []
l_size = [784, 100, 10]
num_layers = len(l_size)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def feedforward(neurons, weights, biases):
	for l in range(1, num_layers):
		neurons[l] = sigmoid(np.dot(weights[l], neurons[l-1].T) + biases[l])
	return neurons

for i in l_size:
	neurons.append(np.full(i, 0.0))

weights = np.load("weights.npz", allow_pickle=True)["arr_0"]
biases = np.load("biases.npz", allow_pickle=True)["arr_0"]

def fillCol(img, i, j, col):
	if i < 0 or i >= img.shape[0] or j < 0 or j >= img.shape[1] or img[i][j] == col or img[i][j] == 0:
		return img, 0

	img[i][j] = col
	img, num1 = fillCol(img, i+1, j, col)
	img, num2 = fillCol(img, i-1, j, col)
	img, num3 = fillCol(img, i, j+1, col)
	img, num4 = fillCol(img, i, j-1, col)
	return img, 1+num1+num2+num3+num4

def shiftImage(img, i, j) :
	img2 = np.full(img.shape, 0.0)
	for a in range(img.shape[0]) :
		for b in range(img.shape[1]) :
			if img[a][b] == 255 :
				img2[a+i][b+j] = 255.0
	return img2

img = cv2.imread('sud2.jpg')
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

pts1 = np.float32([[X[a2]+9, Y[a2]+9], [X[a3]-9, Y[a3]+9], [X[a1]-9, Y[a1]-9], [X[a4]+9, Y[a4]-9]])
pts2 = np.float32([[0,0],[306,0],[306,306],[0,306]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(imgray,M,(306,306))

digits = np.full((9, 9), 0)

for i in range(0, 273, 34):
	for j in range(0, 273, 34):
		cell = dst[i+3:i+31, j+3:j+31]
		ret, img2 = cv2.threshold(cv2.equalizeHist(cell), 35, 255, cv2.THRESH_BINARY_INV)
		ar = 0
		y_m = 0
		x_m = 0
		for y in range(img2.shape[0]):
			for x in range(img2.shape[1]):
				if img2[y][x] == 255:
					img2, num = fillCol(img2, y, x, 120)
					if num > ar:
						ar = num
						y_m = y
						x_m = x
		img2, num_ = fillCol(img2, y_m, x_m, 255)
		for y in range(img2.shape[0]):
			for x in range(img2.shape[1]):
				if img2[y][x] == 120:
					img2, num = fillCol(img2, y, x, 0)

		pps = np.nonzero(img2)
		X_ = pps[1]
		Y_ = pps[0]
		ym = (np.min(Y_) + np.max(Y_))/2
		xm = (np.min(X_) + np.max(X_))/2
		rows,cols = img2.shape
		img2 = shiftImage(img2, int(rows/2-ym), int(cols/2-xm))
		# M = np.float32([[1,0,cols/2-xm],[0,1,rows/2-ym]])
		# img2 = cv2.warpAffine(img2, M, (cols, rows))
		# ret, img2 = cv2.threshold(img2, 120, 255, cv2.THRESH_BINARY)
		# cv2.imshow("image", img2)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		neurons[0] = np.divide(img2[img2 > -1], 255.0)
		neurons = feedforward(neurons, weights, biases)
		digits[int(i/34)][int(j/34)] = np.argmax(neurons[num_layers-1])

print(digits)
