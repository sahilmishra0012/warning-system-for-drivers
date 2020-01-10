import cv2

img=cv2.imread('image.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)   # RGB to YCrCb

img = cv2.medianBlur(img, 3)    # Median Filtering

Y,Cr,Cb = cv2.split(img) # Channel Splitting

# Histogram Equalization of each channel
output_Y = cv2.equalizeHist(Y)
output_Cr = cv2.equalizeHist(Cr)
output_Cb = cv2.equalizeHist(Cb)

img = cv2.merge((output_Y, output_Cr, output_Cb))
img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()