import cv2
class ImagePreprocess:
    def __init__(self,file_name):
        self.file=file_name
        self.img=cv2.imread(file)   # Read Image File

    def rgb_ycrcb():    # RGB to YCrCb
        temp_img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        return temp_img

    def ycrcb_rgb():    # YCrCb to RGB
        temp_img=cv2.cvtColor(img,cv2.COLOR_YCrCb2RGB)
        return temp_img

    def median_blur():
        temp_img = cv2.medianBlur(img, 3)    # Median Filtering
        return temp_img

    def histogram_equalize():
        Y,Cr,Cb = cv2.split(img) # Channel Splitting

        # Histogram Equalization of each channel
        output_Y = cv2.equalizeHist(Y)  # Luma Channel
        output_Cr = cv2.equalizeHist(Cr)  # Red Chroma Channel
        output_Cb = cv2.equalizeHist(Cb)  # Blue Chroma Channel

        temp_img = cv2.merge((output_Y, output_Cr, output_Cb))   # Merge channels\
        return temp_img