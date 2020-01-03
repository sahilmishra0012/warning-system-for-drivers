import xmltodict
import os
import numpy as np
import glob

bboxes = []
classes_raw = []
annotations_paths = glob.glob( '*.xml' )
for xmlfile in annotations_paths:
    x=xmltodict.parse(open( xmlfile , 'rb' ))
    for i in x['annotation']['object']:
        print(i['name'],i['bndbox']['xmin'],i['bndbox']['ymin'],i['bndbox']['xmax'],i['bndbox']['ymax'])
        print("\n")