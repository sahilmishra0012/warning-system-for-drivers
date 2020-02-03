import xmltodict
import glob
import cv2
from dicttoxml import dicttoxml

xml_file = '/home/samthekiller/Desktop/img.xml'
img_file='/home/samthekiller/Desktop/img.jpg'
filename='/home/samthekiller/Desktop/img1.jpg'
x=xmltodict.parse(open( xml_file , 'rb' ))
img=cv2.imread(img_file)
width = int(img.shape[1] * 65 / 100)
height = int(img.shape[0] * 65 / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
for i in x['annotation']['object']:
    if i['name']=='traffic sign':
        Xminvalue=int(int(i['bndbox']['xmin'])* 65 / 100)
        Xmaxvalue=int(int(i['bndbox']['xmax'])* 65 / 100)
        Yminvalue=int(int(i['bndbox']['ymin'])* 65 / 100)
        Ymaxvalue=int(int(i['bndbox']['ymax'])* 65 / 100)
        img = cv2.rectangle(img,(Xminvalue,Yminvalue),(Xmaxvalue,Ymaxvalue),(0,255,0),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img=img[Xminvalue:Yminvalue,Xmaxvalue,Ymaxvalue]
        cv2.imshow('Image Window',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.putText(img,i['name'],(Xminvalue,Yminvalue), font, 0.5,(0,255,0),1,cv2.LINE_AA)
cv2.imshow('Image Window',img)
cv2.waitKey(0)
cv2.imwrite(filename, img) 
cv2.destroyAllWindows()

