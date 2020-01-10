# import numpy as np
# import cv2
# from scipy import ndimage

# from scipy.signal import convolve2d
# from skimage import color, data, restoration
# from skimage.morphology import disk, dilation
#     # x = dilation(x, selem=mask)
# img=cv2.imread('image.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.medianBlur(img, 5)

# psf = disk(5)
# noise_mean=0
# noise_var=0.0001
# estimated_nsr = noise_var / ndimage.variance(img)
# # img += 0.1 * img.std() * np.random.standard_normal(img.shape)
# img = restoration.richardson_lucy(img, estimated_nsr, iterations=30)
# cv2.imshow("Image Window",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()