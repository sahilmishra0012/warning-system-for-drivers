import cv2
import numpy as np

img=cv2.imread('image.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img = cv2.medianBlur(img, 3)

img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
cv2.imshow('Image',img)
cv2.waitKey(0)