from ImagePreprocess import ImagePreprocess
import cv2
img=ImagePreprocess('image.jpg')

temp_img=img.deblur()
cv2.imshow('Image',temp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()