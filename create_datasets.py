import xml.etree.ElementTree as ET
import os

# NEED TO HAVE FIRST RUN ./load_oxfordpets.sh
DATA_DIR = 'data'


# create dataset from bounding boxes


bound_box_dir = os.path.join(DATA_DIR, 'annotations/xmls')
for filename in os.listdir(bound_box_dir):
    tree = ET.parse(os.path.join(bound_box_dir, filename))
    root = tree.getroot()
    bndbox = root[5][4][0]
    print(filename, bndbox.text)






# create dataset from CAM heatmaps
# create dataset from SAM predictions