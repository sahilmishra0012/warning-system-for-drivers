import xmltodict
import glob
import cv2
from dicttoxml import dicttoxml

xml_file = '/home/samthekiller/Desktop/000720_r.xml'
img_file='/home/samthekiller/Desktop/000720_r.jpg'
filename='/home/samthekiller/Desktop/000720_r1.jpg'
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
        # cv2.putText(img,'Speed Limit - 40 Km/Hr',(Xminvalue,Yminvalue), font, 0.5,(0,255,0),1,cv2.LINE_AA)
cv2.imshow('Image Window',img)
cv2.waitKey(0)
cv2.imwrite(filename, img) 
cv2.destroyAllWindows()
print('Speed Limit- 40 Km/Hr')

