import numpy as np
import cv2

from scipy.signal import convolve2d
from skimage import color, data, restoration

img=cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 5)

psf = np.ones((3, 3)) / 9
img = convolve2d(img, psf, 'same')
img += 0.1 * img.std() * np.random.standard_normal(img.shape)
img = restoration.richardson_lucy(img, psf, iterations=30)
cv2.imshow("Image Window",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# % Load image
# I = im2double(imread('IMG_REAL_motion_blur.PNG'));
# figure(1); imshow(I); title('Source image');

# %PSF
# PSF = fspecial('motion', 14, 0);
# noise_mean = 0;
# noise_var = 0.0001;
# estimated_nsr = noise_var / var(I(:));

# I = edgetaper(I, PSF);
# figure(2); imshow(deconvwnr(I, PSF, estimated_nsr)); title('Result');