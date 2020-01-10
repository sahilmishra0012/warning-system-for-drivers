import cv2
import numpy as np
class ImagePreprocess:
    def __init__(self,file_name):
        self.img=cv2.imread(file_name)   # Read Image File

    def deblur(self):  
        temp_img=cv2.cvtColor(self.img,cv2.COLOR_BGR2YCrCb) # RGB to YCrCb
        temp_img = cv2.medianBlur(temp_img, 5)    # Median Filtering
        Y,Cr,Cb = cv2.split(temp_img) # Channel Splitting

        # Histogram Equalization of each channel
        output_Y = cv2.equalizeHist(Y)  # Luma Channel
        output_Cr = cv2.equalizeHist(Cr)  # Red Chroma Channel
        output_Cb = cv2.equalizeHist(Cb)  # Blue Chroma Channel

        temp_img = cv2.merge((output_Y, output_Cr, output_Cb))   # Merge channels
        temp_img=cv2.cvtColor(temp_img,cv2.COLOR_YCrCb2BGR) #   YCrCb to RGB
        return temp_img