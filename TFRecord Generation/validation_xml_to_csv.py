import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
train_files=pd.read_csv('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/data/IDD_Detection/val.txt',sep='/',header=None)

xml_list = []
for xmlfile in tqdm(train_files.iterrows()):
    xml_path='/home/samthekiller/Downloads/Smart India Hackathon/INTEL/data/IDD_Detection/Annotations/'+xmlfile[1][0]+'/'+xmlfile[1][1]+'/'+xmlfile[1][2]+'.xml'
    tree=ET.parse(xml_path)
    root = tree.getroot()
    for member in root.findall('object'):
        value = ('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/data/IDD_Detection/JPEGImages/'+xmlfile[1][0]+'/'+xmlfile[1][1]+'/'+root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[1][0].text),
                int(member[1][1].text),
                int(member[1][2].text),
                int(member[1][3].text)
                )
        if os.path.exists(value[0]):
                xml_list.append(value)
        else:
                continue
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv(('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/data/IDD_Detection/val_labels.csv'),index=None)
