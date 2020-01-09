import cv2
import numpy as np

img=cv2.imread('image.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img = cv2.medianBlur(img, 3)

R, G, B = cv2.split(img)

output1_R = cv2.equalizeHist(R)
output1_G = cv2.equalizeHist(G)
output1_B = cv2.equalizeHist(B)

img = cv2.merge((output1_R, output1_G, output1_B))

img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
cv2.imshow('Image',img)
cv2.waitKey(0)