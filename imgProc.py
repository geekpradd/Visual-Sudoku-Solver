import cv2
import numpy as np
#import subprocess

img = cv2.imread('sud.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray,100,200)
lines = cv2.HoughLines(edges,1,np.pi/180, 150) 
print(lines)
  
# The below for loop runs till r and theta values  
# are in the range of the 2d array
ky = imgray.shape[0]*np.sqrt(2)
kx = imgray.shape[1]*np.sqrt(2)

for i in lines:
	r = i[0][0]
	theta = i[0][1]
	# Stores the value of cos(theta) in a 
	a = np.cos(theta) 
  
	# Stores the value of sin(theta) in b 
	b = np.sin(theta) 
	  
	# x0 stores the value rcos(theta) 
	x0 = a*r 
	  
	# y0 stores the value rsin(theta) 
	y0 = b*r 
	  
	# x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
	x1 = int(x0 + kx*(-b)) 
	  
	# y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
	y1 = int(y0 + ky*(a)) 
  
	# x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
	x2 = int(x0 - kx*(-b)) 
	  
	# y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
	y2 = int(y0 - ky*(a)) 
	  
	#cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
	#(0,0,255) denotes the colour of the line to be  
	#drawn. In this case, it is red.  
	cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)

mean = 0

for i in range(imgray.shape[0]) :
	first = False
	first_coord = 0
	last_coord = 0
	for j in range(imgray.shape[1]) :
		x = imgray.item(i, j);
		if x < 120 :
			if not first:
				first_coord = j
				first = True
			last_coord = j
	mean += last_coord - first_coord
mean = mean/(9*img.shape[0])

cellCords = []
temp = []
for i in range(9) :
	temp.append(0)
for i in range(9) :
	cellCords.append(temp)



# for i in range(imgray.shape[0]) :
# 	for j in range(imgray.shape[1]) :
# 		x = imgray.item(i, j)
# 		if x > 120 :
# 			img[i, j] = [255, 255, 255]
# 			imgray[i, j] = 255
# 		else :
# 			img[i, j] = [0, 0, 0]
# 			imgray[i, j] = 0

# contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# maxP = cv2.arcLength(contours[0], True)
# max_i = 0
# for i in range(1, len(contours)) :
# 	perimeter = cv2.arcLength(contours[i], True)
# 	if perimeter > maxP :
# 		maxP = perimeter
# 		max_i = i

# cv2.drawContours(img, contours, max_i, (0,255,0), 3)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()