import cv2

class ImagePreprocess:
    def __init__(self,file_name):
        self.file=file_name
        self.img=cv2.imread(file)

    def rgb_ycrcb(file):
        temp_img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        return temp_img

    def ycrcb_rgb(file):
        temp_img=cv2.cvtColor(img,cv2.COLOR_YCrCb2RGB)
        return temp_img

    def ycrcb_rgb(file):
        temp_img=cv2.cvtColor(img,cv2.COLOR_YCrCb2RGB)
        return temp_img

img=cv2.imread('image.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)   # RGB to YCrCb

img = cv2.medianBlur(img, 3)    # Median Filtering

Y,Cr,Cb = cv2.split(img) # Channel Splitting

# Histogram Equalization of each channel
output_Y = cv2.equalizeHist(Y)  # Luma Channel
output_Cr = cv2.equalizeHist(Cr)  # Red Chroma Channel
output_Cb = cv2.equalizeHist(Cb)  # Blue Chroma Channel

img = cv2.merge((output_Y, output_Cr, output_Cb))   # Merge channels

img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)   # YCrCb to RGB
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()