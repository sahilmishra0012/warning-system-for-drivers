import numpy as np
import cv2

from scipy.signal import convolve2d
from skimage import color, data, restoration

from skimage import color, data, restoration,io

img = io.imread('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/Warning System for Drivers/Color Extraction/Data/0003660.jpg')
img=color.rgb2gray(img)
psf = np.ones((5, 5)) / 25
img = convolve2d(img, psf, 'same')
astro_noisy = img.copy()

astro_noisy += (np.random.poisson(lam=25, size=img.shape) - 10) / 255

deconv = restoration.richardson_lucy(astro_noisy, psf,30)
deconv=color.gray2rgb(deconv)
cv2.imshow("Image Window",deconv)
cv2.waitKey(0)
cv2.destroyAllWindows()