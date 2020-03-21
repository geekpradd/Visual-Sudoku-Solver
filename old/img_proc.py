import cv2
import numpy as np 
from digit_read import get_digit

print("yo")
img = cv2.imread("pic.png")
img = cv2.resize(img, (700, 700), interpolation=cv2.INTER_AREA)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


for i in range(imggray.shape[0]) :
 	for j in range(imggray.shape[1]) :
 		x = imggray.item(i, j)
 		if x > 120:
 			img[i, j] = [0, 0, 0]
 			imggray[i, j] = 0
 		else :
 			img[i, j] = [255, 255, 255]
 			imggray[i, j] = 255

             
contours, hier = cv2.findContours(imggray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_perim_i = np.argmax([cv2.arcLength(contour, True) for contour in contours])
img = cv2.drawContours(img, contours, max_perim_i, (1, 0, 0), 3)

l_y = 700
m_y = 0
points = []
for i in range(imggray.shape[0]) :
 	for j in range(imggray.shape[1]) :
         if img[i, j][0] == 1 and img[i, j][1] == 0 and img[i,j][2]==0:
             m_y = max(m_y, i)
             l_y = min(l_y, i)
             points.append([i, j])

# this is for bottom area
least_x_b = 700
y_for_least_x_b = 0
max_x_b = 0
y_for_max_x_b = 0
least_x_t = 700
y_for_least_x_t = 0
max_x_t = 0
y_for_max_x_t = 0

for i, j in points:
        if abs(i-m_y)<=20:
            if least_x_b > j:
                least_x_b = j
                y_for_least_x_b = i
            if max_x_b < j:
                max_x_b = j
                y_for_max_x_b = i
        if abs(i-l_y)<=20:
            if least_x_t > j:
                least_x_t = j
                y_for_least_x_t = i 
            if max_x_t < j:
                max_x_t = j
                y_for_max_x_t = i


initial = np.float32([[least_x_t, y_for_least_x_t,],[max_x_t, y_for_max_x_t],[max_x_b, y_for_max_x_b],[least_x_b, y_for_least_x_b]])
final = np.float32([[0,0], [630, 0], [630, 630], [0, 630]])

M = cv2.getPerspectiveTransform(initial, final)

img_f = cv2.warpPerspective(img, M, (630, 630))
# ret, img_f_b = cv2.threshold(img_f,127,255,cv2.THRESH_BINARY)
# img_f_b = cv2.cvtColor(img_f_b, cv2.COLOR_BGR2GRAY)

# hough = cv2.HoughLines(img_f_b,1,np.pi/180, 400)
# print (len(hough))

# for p, th in hough[0]:
#     cos = np.cos(th)
#     sin = np.sin(th)
#     x1 = int(p*cos - 500*sin)
#     y1 = int(p*sin + 500*cos)
#     x2 = int(p*cos + 500*sin)
#     y2 = int(p*sin - 500*cos)

#     cv2.line(img_f, (x1, y1), (x2, y2), (255, 0, 0), 3)
# perform hough transform on img_f
# canny = 

# print(least_x_b, y_for_least_x_b)
# print(max_x_b, y_for_max_x_b)
# print(least_x_t, y_for_least_x_t)
# print(max_x_t, y_for_max_x_t)
roi = img_f[70*0:70*1, 70*4:70*5]
print (get_digit(roi))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 630, 630)
cv2.imshow('image', img_f)
cv2.waitKey(0)
cv2.destroyAllWindows()