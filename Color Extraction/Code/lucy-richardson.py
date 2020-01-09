import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, restoration




PSF = fspecial('disk', 8);
noise_mean = 0;
noise_var = 0.0001;
estimated_nsr = noise_var / var(I(:));

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)