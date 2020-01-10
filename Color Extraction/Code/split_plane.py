import cv2
import numpy as np
img = cv2.imread("image.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

y,cr,cb = cv2.split(img)
cv2.imshow("Image Window",cr)
cv2.waitKey(0)
cv2.destroyAllWindows()