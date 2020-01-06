import cv2

img=cv2.imread('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/Warning System for Drivers/Color Extraction/Data/0003660.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img = cv2.medianBlur(img, 5)
cv2.imshow('Image',img)
cv2.waitKey(0)