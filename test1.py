import xmltodict
import os
import numpy as np
import glob
from PIL import Image
from PIL import ImageDraw
import cv2

annotations_paths = glob.glob('*.xml')
img_paths=glob.glob('*.jpg')
for xmlfile,imgfile in zip(annotations_paths,img_paths):
    x=xmltodict.parse(open( xmlfile , 'rb' ))
    img=cv2.imread(imgfile)
    for i in x['annotation']['object']:
        Xminvalue=int(i['bndbox']['xmin'])
        Xmaxvalue=int(i['bndbox']['xmax'])
        Yminvalue=int(i['bndbox']['ymin'])
        Ymaxvalue=int(i['bndbox']['ymax'])
        name=i['name']
        img = cv2.rectangle(img,(Xminvalue,Yminvalue),(Xmaxvalue,Ymaxvalue),(0,255,0),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,name,(Xminvalue,Yminvalue), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    cv2.imshow('Image Window',img)
    k=cv2.waitKey(400)