from ImagePreprocess import ImagePreprocess
import cv2
img=ImagePreprocess('image.jpg')
temp_img=img.deblur()
Y,Cr,Cb = cv2.split(temp_img) # Channel Splitting
ret, thresh2 = cv2.threshold(Cr, 80, 255, cv2.THRESH_BINARY_INV)
template=cv2.imread('temp.jpg')
cv2.imshow('Image',thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)