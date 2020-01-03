import xmltodict
import os
import numpy as np
import glob

bboxes = []
classes_raw = []
annotations_paths = glob.glob( '/.xml' )
for xmlfile in annotations_paths:
    x = xmltodict.parse( open( xmlfile , 'rb' ) )
    bndbox = x[ 'annotation' ][ 'object' ][ 'bndbox' ]
    bndbox = np.array([ int(bndbox[ 'xmin' ]) , int(bndbox[ 'ymin' ]) , int(bndbox[ 'xmax' ]) , int(bndbox[ 'ymax' ]) ])
    bndbox2 = [ None ] * 4
    bndbox2[0] = bndbox[0]
    bndbox2[1] = bndbox[1]
    bndbox2[2] = bndbox[2]
    bndbox2[3] = bndbox[3]
    bndbox2 = np.array( bndbox2 )
    bboxes.append( bndbox2 )
    classes_raw.append( x[ 'annotation' ][ 'object' ][ 'name' ] )