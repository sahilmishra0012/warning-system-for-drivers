import cv2
import numpy as np
img = cv2.imread("/home/samthekiller/Downloads/Smart India Hackathon/INTEL/Warning System for Drivers/Test Code/img1.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

Y,Cr,Cb = cv2.split(img)

output_Y = cv2.equalizeHist(Y)
output_Cr = cv2.equalizeHist(Cr)
output_Cb = cv2.equalizeHist(Cb)
img = cv2.merge((output_Y, output_Cr, output_Cb))

ret, thresh2 = cv2.threshold(output_Cr, 200, 255, cv2.THRESH_BINARY) 

cv2.imshow('Image',thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()