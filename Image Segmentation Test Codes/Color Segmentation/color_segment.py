from ImagePreprocess import ImagePreprocess
import cv2
img=ImagePreprocess('image.jpg')
temp_img=img.deblur()
Y,Cr,Cb = cv2.split(temp_img) # Channel Splitting
ret, thresh2 = cv2.threshold(Cr, 80, 255, cv2.THRESH_BINARY_INV) 

cv2.imshow('Image',thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()