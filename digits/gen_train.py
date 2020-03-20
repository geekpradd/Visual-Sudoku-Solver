import cv2 
import numpy as np 

def get_stretched(img, p, q):
    resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
    regray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mean = np.sum(regray)/(4*50*50)

    ret, regray = cv2.threshold(regray, 100, 255, cv2.THRESH_BINARY_INV)
    # regray = cv2.Threshold(regray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY_INV,5,2)digit

    
    m_y = 1000
    x_m_y = -1
    m_x = 1000
    y_m_x = -1
    max_y = 0
    x_max_y = -1 
    max_x = 0
    y_max_x = -1

    for y in range(50):
        for x in range(50):
            if regray[y, x] == 255:
                if m_y > y:
                    m_y = y 
                    x_m_y = x 
                if m_x > x:
                    m_x = x 
                    y_m_x = y 
                if max_y < y:
                    max_y = y 
                    x_max_y = x
                if max_x < x:
                    max_x = x
                    y_max_x = y

    c_x = (m_x + max_x)/2
    c_y = (m_y + max_y)/2
    area = (max_x - m_x)*(max_y - m_y) 
    roi = regray[m_y:max_y, m_x:max_x] 
    
    gap = min(m_x, 50 - max_x + 1, m_y, 50 - max_y + 1)
    # can change by gap 
    m_x_f = int(m_x - p*gap/q)
    max_x_f = int(max_x + p*gap/q)
    m_y_f = int(m_y - p*gap/q)
    max_y_f = int(max_y + p*gap/q)
    roi_f = cv2.resize(roi, (max_x_f - m_x_f, max_y_f - m_y_f) , interpolation=cv2.INTER_AREA)

    img_blank = np.zeros((50, 50))
    for x in range(m_x_f, max_x_f):
        for y in range(m_y_f, max_y_f):
            # print ("got " + str(x - m_x_f) + " and " + str(y - m_y_f))
            img_blank[y, x] = roi_f[y - m_y_f,   x - m_x_f]

    # cv2.imshow("image", img_blank)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_blank

# img = cv2.imread('digits/q'+str(5)+'.jpg')
# # resized = cv2.resize(img, (50, 50))
# # ar = np.subtract(255, resized)

# # print(ar)
# i1 = cv2.resize(get_stretched(img, 9, 10), (28, 28), interpolation=cv2.INTER_AREA)
# # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
# # i1 = cv2.erode(i1, kernel, iterations = 1)
# # kernel = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
# # i1 = cv2.dilate(i1, kernel, iterations = 1)

# cv2.imshow("image", i1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
