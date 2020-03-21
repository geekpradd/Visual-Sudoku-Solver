import cv2 
import numpy as np 

def get_digit(img):
    resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
    regray = cv2.equalizeHist(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    regray = cv2.GaussianBlur(regray, (11, 11), 0)
    mean = np.sum(regray)/(4*50*50)
    
    cv2.imshow("image", regray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret, regray = cv2.threshold(regray, mean, 255, cv2.THRESH_BINARY_INV)
    # regray = cv2.Threshold(regray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY_INV,5,2)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 100, 100)
    cv2.imshow('image', regray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
    # pre process code here to shift and blur stuff out 
    min_val = 10000000
    dig = -1

    
    for digit in range(1, 10):
        target = cv2.imread("base/d" + str(digit) +  ".png")
        print (target.shape)
        target = cv2.resize(target, (50, 50), interpolation=cv2.INTER_AREA)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.GaussianBlur(target_gray, (11, 11), 0)

        target_gray = cv2.adaptiveThreshold(target_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,5,2)
        

        diff = np.sum(np.abs(np.subtract(target_gray,regray)))

        print ("For digit " + str(digit) + " got " + str(diff))
        if diff < min_val:
            dig = digit
            min_val = diff

    print(dig)

img = cv2.imread("digits/p2.jpg")
print (img.shape)
get_digit(img)

