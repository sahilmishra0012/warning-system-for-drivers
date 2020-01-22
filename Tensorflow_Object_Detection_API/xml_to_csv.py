import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
xml_list = []
path = '/home/samthekiller/Downloads/Smart India Hackathon/INTEL/Warning System for Drivers/data'

annotations_paths = [f for f in glob.glob(path + "/**/*.xml", recursive=True)]


for xmlfile in sorted(annotations_paths):
    tree=ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
                )
        xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)

