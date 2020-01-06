import pandas as pd
import glob
import cv2

data=pd.read_csv('final_csv.csv')
path='/home/samthekiller/Downloads/Smart India Hackathon/INTEL/signDatabasePublicFramesOnly/'
for i in data.iterrows():
    Xminvalue=int(i[1]['Lower right corner X'])
    Xmaxvalue=int(i[1]['Upper left corner X'])
    Yminvalue=int(i[1]['Lower right corner Y'])
    Ymaxvalue=int(i[1]['Upper left corner Y'])
    name=i[1]['Filename']
    annot_name=i[1]['Annotation tag']
    img=cv2.imread(path+name)
    img
    img = cv2.rectangle(img,(Xmaxvalue,Ymaxvalue),(Xminvalue,Yminvalue),(255,255,0),1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,annot_name,(Xminvalue,Yminvalue), font, 0.5,(255,255,0),1,cv2.LINE_AA)
    cv2.imshow('Image Window',img)
    k=cv2.waitKey(1000)