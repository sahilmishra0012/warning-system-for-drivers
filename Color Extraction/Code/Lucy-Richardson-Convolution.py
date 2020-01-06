import numpy as np
import cv2

from scipy.signal import convolve2d
from skimage import color, data, restoration

img=cv2.imread('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/Warning System for Drivers/Color Extraction/Data/0003660.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 5)

psf = np.ones((5, 5)) / 25
img = convolve2d(img, psf, 'same')
img += 0.1 * img.std() * np.random.standard_normal(img.shape)
img = restoration.richardson_lucy(img, psf, iterations=30)
cv2.imshow("Image Window",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

