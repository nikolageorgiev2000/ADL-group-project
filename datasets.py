import xml.etree.ElementTree as ET
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchvision.datasets import OxfordIIITPet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, recall_score, jaccard_score, f1_score
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Set

print("NEED TO HAVE FIRST RUN:\n./load_oxfordpets.sh")

# Set up the data directories
DATA_DIR = 'data/images'
ANNOTATION_DIR = 'data/annotations'

DatasetSelection = Enum('DatasetSelection', [('Trimap', 1), ('Class', 2), ('BBox', 3), ('CAM', 4), ('SAM', 5)])

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Define transformations for training
BASE_2D_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(device).squeeze()),
])

IMAGE_TRANSFORM = transforms.Compose([
    BASE_2D_TRANSFORM,
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.to(torch.float16)),
])

TRIMAP_TRANSFORM = transforms.Compose([
    BASE_2D_TRANSFORM,
    transforms.Lambda(lambda x: x.to(torch.int8)),
])

class InMemoryPetSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, targets_list: List[DatasetSelection], image_transform=IMAGE_TRANSFORM, trimap_transform=TRIMAP_TRANSFORM, image_shape=(224, 224)):
        available_targets = set(DatasetSelection)
        assert set(targets_list).issubset(available_targets)
        self.targets_list = targets_list

        self.image_transform = image_transform
        self.trimap_transform = trimap_transform
        self.samples = []

        # get image filenames
        image_files = [f for f in os.listdir(
            data_dir) if f.endswith(('.jpg', '.png'))]
        self.image_ind_dict = {
            f.split('.')[0]: i for i, f in enumerate(image_files)}

        # get image classes
        contents = np.genfromtxt(os.path.join(annotation_dir, 'list.txt'), skip_header=6, usecols=(
            0, 1), dtype=[('name', np.str_, 32), ('grades', np.uint8)])
        # check all animals are present in dict keys
        self.labels_dict = {str(x[0]): int(x[1] - 1) for x in contents}
        assert len(contents) == len(self.labels_dict.keys())

        # get bounding boxes
        xml_dir = os.path.join(annotation_dir, 'xmls')
        self.bbbox_dict = {}
        for filename in os.listdir(xml_dir):
            tree = ET.parse(os.path.join(xml_dir, filename))
            root = tree.getroot()
            # xmin, ymin, xmax, ymax
            xmin, ymin, xmax, ymax = (
                int(root[5][4][i].text) for i in range(4))
            width, height = int(root[3][0].text), int(root[3][1].text)
            self.bbbox_dict[root[1].text.split('.')[0]] = np.array([
                xmin * image_shape[0] / width,
                ymin * image_shape[1] / height,
                xmax * image_shape[0] / width,
                ymax * image_shape[1] / height,
            ]).astype(int)

        # get subset of images that have all target annotations available
        self.available_images = set(self.image_ind_dict.keys())
        if DatasetSelection.Class in targets_list:
            self.available_images.intersection_update(self.labels_dict.keys())

        if DatasetSelection.BBox in targets_list:
            self.available_images.intersection_update(self.bbbox_dict.keys())
        print(f'available samples: {self.__len__()}')

        # self.available_images = list(self.available_images)[:500]
        for fname in tqdm.tqdm(self.available_images):
            fname_with_extension = fname + '.jpg'

            # Load image
            img_path = os.path.join(data_dir, fname_with_extension)
            img = Image.open(img_path).convert('RGB')
            img = self.image_transform(img)

            # Load and preprocess trimap
            # ensure file extension matches annotation
            trimap_file = fname_with_extension.replace('.jpg', '.png')
            trimap_path = os.path.join(annotation_dir, 'trimaps', trimap_file)
            trimap = Image.open(trimap_path)
            trimap = self.trimap_transform(trimap)
            trimap[trimap == 1] = 0
            trimap[trimap == 2] = 1
            trimap[trimap == 3] = 2

            # Save original image (not transformed yet) and processed trimap
            sample_data = {}
            if DatasetSelection.Trimap in self.targets_list:
                sample_data[DatasetSelection.Trimap] = trimap
            if DatasetSelection.Class in self.targets_list:
                sample_data[DatasetSelection.Class] = self.labels_dict[fname]
            if DatasetSelection.BBox in self.targets_list:
                sample_data[DatasetSelection.BBox] = self.bbbox_dict[fname]
            self.samples.append((img, sample_data))



    def __len__(self):
        return len(self.available_images)

    def __getitem__(self, idx):
        return self.samples[idx]


InMemoryPetSegmentationDataset(DATA_DIR, ANNOTATION_DIR, [DatasetSelection.BBox])