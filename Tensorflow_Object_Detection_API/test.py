import xmltodict
import glob
import cv2


path = '/home/samthekiller/Downloads/Smart India Hackathon/INTEL/clean/IDD_Detection/data'

annotations_paths = [f for f in glob.glob(path + "/**/*.xml", recursive=True)]

img_paths = [f for f in glob.glob(path + "/**/*.jpg", recursive=True)]

for xmlfile,imgfile in zip(sorted(annotations_paths),sorted(img_paths)):
    x=xmltodict.parse(open( xmlfile , 'rb' ))
    img=cv2.imread(imgfile)
    width = int(img.shape[1] * 65 / 100)
    height = int(img.shape[0] * 65 / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    try:
        for i in x['annotation']['object']:
            Xminvalue=int(int(i['bndbox']['xmin'])* 65 / 100)
            Xmaxvalue=int(int(i['bndbox']['xmax'])* 65 / 100)
            Yminvalue=int(int(i['bndbox']['ymin'])* 65 / 100)
            Ymaxvalue=int(int(i['bndbox']['ymax'])* 65 / 100)
            name=i['name']
            img = cv2.rectangle(img,(Xminvalue,Yminvalue),(Xmaxvalue,Ymaxvalue),(255,255,0),1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,name,(Xminvalue,Yminvalue), font, 0.5,(255,255,0),1,cv2.LINE_AA)
        cv2.imshow('Image Window',img)
        k=cv2.waitKey(30)
    except:
        pass