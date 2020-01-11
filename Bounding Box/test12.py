import xmltodict
import glob
import cv2
from dicttoxml import dicttoxml

annotations_paths = glob.glob('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/clean/IDD_Detection/Data/**/**/*.xml')
img_paths=glob.glob('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/clean/IDD_Detection/Data/**/**/*.jpg')
for xmlfile,imgfile in zip(sorted(annotations_paths,reverse=True),sorted(img_paths,reverse=True)):
    x=xmltodict.parse(open( xmlfile , 'rb' ))
    try:
        for i in x['annotation']['object']:
            if i['name']=='traffic sign':
                img=cv2.imread(imgfile)
                width = int(img.shape[1] * 65 / 100)
                height = int(img.shape[0] * 65 / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                Xminvalue=int(int(i['bndbox']['xmin'])* 65 / 100)
                Xmaxvalue=int(int(i['bndbox']['xmax'])* 65 / 100)
                Yminvalue=int(int(i['bndbox']['ymin'])* 65 / 100)
                Ymaxvalue=int(int(i['bndbox']['ymax'])* 65 / 100)
                img = cv2.rectangle(img,(Xminvalue,Yminvalue),(Xmaxvalue,Ymaxvalue),(255,255,0),1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,i['name'],(Xminvalue,Yminvalue), font, 0.5,(255,255,0),1,cv2.LINE_AA)
        cv2.imshow('Image Window',img)
        cv2.waitKey(30)
    except:
        pass
cv2.destroyAllWindows()

