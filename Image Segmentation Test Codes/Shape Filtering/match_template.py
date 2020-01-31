from ImagePreprocess import ImagePreprocess
import cv2
import numpy as np
immg=cv2.imread('image1.jpg')
img=ImagePreprocess('image1.jpg')
temp_img=img.deblur()
Y,Cr,Cb = cv2.split(temp_img) # Channel Splitting
ret, thresh2 = cv2.threshold(Cr, 80, 255, cv2.THRESH_BINARY)
template=cv2.imread('images.jpeg')
template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]
print(template.shape)
res = cv2.matchTemplate(thresh2,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.113
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(immg, pt, (pt[0] + w, pt[1] + h), (255,255,255), 0)
cv2.imshow('Image',immg)
cv2.waitKey(0)
cv2.destroyAllWindows()