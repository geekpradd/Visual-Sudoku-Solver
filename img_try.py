import cv2
import numpy as np 

img = np.zeros((512, 512, 3), np.uint8)

img = cv2.line(img, (0, 0), (255, 255), (23, 45, 12), 10)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()