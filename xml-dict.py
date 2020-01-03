import xmltodict
import os
import numpy as np
import glob
from PIL import Image
from PIL import ImageDraw
import cv2

img=cv2.imread('0001999.jpg')
annotations_paths = glob.glob( '*.xml' )
for xmlfile in annotations_paths:
    x=xmltodict.parse(open( xmlfile , 'rb' ))
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
    k=cv2.waitKey(0)

    if k==27:#  27 is for escape character
        cv2.destroyAllWindows()# To destroy all the windows
    # Write Image when s is pressed.
    elif k==ord('s'):
        cv2.imwrite('bird.jpg',img)
        cv2.destroyAllWindows()# To destroy all the windows