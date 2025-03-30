import xml.etree.ElementTree as ET
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


# NEED TO HAVE FIRST RUN ./load_oxfordpets.sh
DATA_DIR = 'data'


# create dataset from bounding boxes

def get_dataset(data_dir, annotation_dir, device):
    # Define transformations for training
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device).squeeze()),
    ])

    data_transform = transforms.Compose([
        base_transform,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.to(torch.float32)),
    ])

    target_transform = transforms.Compose([
        base_transform,
        transforms.Lambda(lambda x: x.to(torch.long)),
    ])

class InMemoryPetSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        image_files = [f for f in os.listdir(
            data_dir) if f.endswith(('.jpg', '.png'))]

        for fname in tqdm.tqdm(image_files[:1_000]):
            # Load image
            img_path = os.path.join(data_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img).to(device)

            # Load and preprocess trimap
            # ensure file extension matches annotation
            trimap_file = fname.replace('.jpg', '.png')
            trimap_path = os.path.join(annotation_dir, 'trimaps', trimap_file)
            trimap = Image.open(trimap_path)
            trimap = self.target_transform(trimap)
            trimap[trimap == 1] = 0
            trimap[trimap == 2] = 1
            trimap[trimap == 3] = 2

            # Save original image (not transformed yet) and processed trimap
            self.samples.append((img, trimap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, trimap = self.samples[idx]
        return img, trimap

bound_box_dir = os.path.join(DATA_DIR, 'annotations/xmls')
for filename in os.listdir(bound_box_dir):
    tree = ET.parse(os.path.join(bound_box_dir, filename))
    root = tree.getroot()
    bndbox = root[5][1]#[0]
    print(filename, bndbox.text)





# create dataset from CAM heatmaps
# create dataset from SAM predictions