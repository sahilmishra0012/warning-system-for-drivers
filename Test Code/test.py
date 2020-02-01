import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math

def constrastLimit(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    img = cv2.medianBlur(img, 3)
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    img = cv2.merge((output1_R, output1_G, output1_B))
    img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
    return img

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    thresh = cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV)[1]
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

def removeSmallComponents(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnts

def contourIsSign(perimeter, centroid, threshold):
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:
        return True, max_value + 2
    else: 
        return False, max_value + 2

def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    return image[top:bottom,left:right]


def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours[0]:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign = cropSign(image,coordinate)
    return coordinate,sign


def findSigns(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours[0]:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle):
    original_image = image.copy()
    binary_image = preprocess_image(image)
    binary_image = removeSmallComponents(binary_image, min_size_components)
    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))
    cv2.imshow('BINARY IMAGE', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours = findContour(binary_image)
    coordinate,sign = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    return coordinate, original_image

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray,10,255,apertureSize = 7)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    return mask

def main():
    frame=cv2.imread('img1.jpg')

    similitary_contour_with_circle = 0.60
    min_size_components=200

    width = frame.shape[1]
    height = frame.shape[0]
    dim = (width, height) 
    frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

    coordinate, image = localization(frame, min_size_components, similitary_contour_with_circle)
    width = image.shape[1]
    height = image.shape[0]
    dim = (width, height)

    try:

        a=coordinate[0][0]
        b=coordinate[0][1]
        c=coordinate[1][0]
        d=coordinate[1][1]

        if coordinate[0][0]<0:
            a=0
        if coordinate[0][1]<0:
            b=0
        if coordinate[1][0]<0:
            c=0
        if coordinate[1][1]<0:
            d=0
        
        if coordinate[0][0]>width:
            a=width-1
        if coordinate[0][1]>height:
            b=height-1
        if coordinate[1][0]>width:
            c=width-1
        if coordinate[1][1]>height:
            d=height-1
        image=image[b:d,a:c]
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print('No Sign Found')
        pass

if __name__ == '__main__':
    main()