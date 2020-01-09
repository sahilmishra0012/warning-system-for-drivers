import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration,io

astro = cv2.imread('image.jpg')
astro=cv2.cvtColor(astro,cv2.COLOR_BGR2GRAY)

psf = np.ones((3, 3)) /9
astro = conv2(astro, psf, 'same')
# Add Noise to Image
astro_noisy = astro.copy()

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=110)
cv2.imshow('Image',deconvolved_RL)
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