import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
xml_list = []
path = '/home/samthekiller/Downloads/Smart India Hackathon/INTEL/data/IDD_Detection/Data/Data1'

annotations_paths = [f for f in glob.glob(path + "/**/**/*.xml", recursive=True)]
xml_list = []
for xmlfile in tqdm(sorted(annotations_paths)):
    tree=ET.parse(xmlfile)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (xmlfile[:-3]+"jpg",
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[1][0].text),
                int(member[1][1].text),
                int(member[1][2].text),
                int(member[1][3].text)
                )
        xml_list.append(value)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_hdf(('/home/samthekiller/Downloads/Smart India Hackathon/INTEL/data/IDD_Detection/Data/labels1.h5'),key='df', mode='w')