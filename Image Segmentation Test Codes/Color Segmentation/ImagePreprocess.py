import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils

def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    ret,thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    image = removeSmallComponents(image,1700)
    return image

def main():
    immg=cv2.imread('image1.jpg')
    image=cv2.imread('image1.jpg')
    image=preprocess_image(image)
    template=cv2.imread('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/Warning System for Drivers/Image Segmentation Test Codes/Shape Filtering/images.jpeg')
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    ret,temp = cv2.threshold(template,32,255,cv2.THRESH_BINARY_INV)
    res = cv2.matchTemplate(image,temp,cv2.TM_CCOEFF_NORMED)
    threshold = np.mean(res)+50*np.var(res)
    w, h = template.shape[::-1]

    loc = np.where( res >= threshold)
    ll=list(zip(*loc[::-1]))
    pt=max(ll,key = ll.count)
    print(pt)

    cv2.rectangle(immg, pt, (pt[0] + w, pt[1] + h), (255,255,255), 0)
    cv2.imshow('Image',immg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()