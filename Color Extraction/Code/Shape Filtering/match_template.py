from ImagePreprocess import ImagePreprocess
import cv2
import numpy as np
img=ImagePreprocess('image.jpg')
temp_img=img.deblur()
Y,Cr,Cb = cv2.split(temp_img) # Channel Splitting
ret, thresh2 = cv2.threshold(Cr, 80, 255, cv2.THRESH_BINARY_INV)
template=cv2.imread('temp.jpg')
template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(thresh2,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(thresh2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv2.imshow('Image',template)
cv2.waitKey(0)
cv2.destroyAllWindows()