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

DatasetSelection = Enum('DatasetSelection', [(
    'Trimap', 1), ('Class', 2), ('BBox', 3), ('CAM', 4)])

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Define transformations for training
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.to(torch.half)),
    transforms.Lambda(lambda x: x.to(device).squeeze()),
])

TRIMAP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.int8)),
    transforms.Lambda(lambda x: x.to(device).squeeze()),
])


class InMemoryPetSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, targets_list: List[DatasetSelection], image_transform=IMAGE_TRANSFORM, trimap_transform=TRIMAP_TRANSFORM, image_shape=(224, 224), training=False):
        self.training = training

        available_targets = set(DatasetSelection)
        assert set(targets_list).issubset(available_targets)
        self.targets_list = targets_list

        self.data_dir = data_dir
        self.annotation_dir = annotation_dir

        self.image_transform = image_transform
        self.trimap_transform = trimap_transform
        self.samples = []

        # get image filenames
        image_files = [f for f in os.listdir(
            data_dir) if f.endswith(('.jpg', '.png'))]
        self.image_ind_dict = {
            f.split('.')[0]: i for i, f in enumerate(image_files)}
        # convert to list to give it an ordering
        self.available_images: List[str] = sorted(self.image_ind_dict.keys())
        print(f'available samples: {self.__len__()}')
        # shuffle
        dataset_permutation = torch.randperm(len(self.available_images))
        self.available_images = [self.available_images[i]
                                 for i in dataset_permutation]  # [:100]
        self.selected_trimap_inds: Set[int] = set(self.available_images)
        self.masking_permutation = torch.randperm(len(self.available_images))

        # get image classes
        contents = np.genfromtxt(os.path.join(annotation_dir, 'list.txt'), skip_header=6, usecols=(
            0, 1), dtype=[('name', np.str_, 32), ('grades', np.uint8)])
        # check all animals are present in dict keys
        self.labels_dict = {str(x[0]): int(x[1] - 1) for x in contents}
        assert all(map(lambda v: 0 <= v < 37, self.labels_dict.values()))
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

        for fname in tqdm.tqdm(self.available_images, disable=True):
            fname_with_extension = fname + '.jpg'

            # Load image
            img_path = os.path.join(data_dir, fname_with_extension)
            img = Image.open(img_path).convert('RGB')
            img = self.image_transform(img)

            sample_data = {}
            if DatasetSelection.Trimap in self.targets_list:
                trimap = self.load_trimap(fname)
                sample_data[DatasetSelection.Trimap] = trimap
            if DatasetSelection.Class in self.targets_list:
                sample_data[DatasetSelection.Class] = self.labels_dict.get(
                    fname, -100)  # ignore index if label missing
            #if DatasetSelection.BBox in self.targets_list:
            #    sample_data[DatasetSelection.BBox] = self.bbbox_dict.get(
            #        fname, -1*np.ones(4, dtype=int))

            if DatasetSelection.BBox in self.targets_list:
                bbox = self.bbbox_dict.get(fname, -1*np.ones(4, dtype=int))
                sample_data[DatasetSelection.BBox] = torch.tensor(bbox, dtype=torch.int)

            if DatasetSelection.CAM in self.targets_list:
                cam_path = os.path.join(annotation_dir, 'heatmaps', fname)
                sample_data[DatasetSelection.CAM] = (torch.load(
                    cam_path, weights_only=False) if os.path.exists(cam_path) else -torch.ones((7, 7), dtype=torch.int8)).to(device)

            self.samples.append((img, sample_data))

        self.dummy_trimap = -100 * \
            torch.ones(image_shape, device=img.device, dtype=torch.int8)

    def load_trimap(self, fname):
        # All images have trimaps
        trimap_file = fname + '.png'
        trimap_path = os.path.join(
            self.annotation_dir, 'trimaps', trimap_file)
        trimap = Image.open(trimap_path)
        trimap = self.trimap_transform(trimap)
        trimap[trimap == 1] = -100  # unknown
        trimap[trimap == 2] = 0     # background
        trimap[trimap == 3] = 1     # foreground
        return trimap

    def __len__(self):
        return len(self.available_images)

    def change_gt_proportion(self, gt_proportion):
        assert 0 <= gt_proportion <= 1.0
        new_selected_trimap_inds = set(
            range(int(gt_proportion * len(self.available_images))))
        for idx in new_selected_trimap_inds:
            if idx not in self.selected_trimap_inds:
                trimap = self.load_trimap(self.available_images[idx])
                img, image_data = self.samples[idx]
                image_data[DatasetSelection.Trimap] = trimap
                self.samples[idx] = (img, image_data)
        for idx in range(len(self.available_images)):
            img, image_data = self.samples[idx]
            if idx in new_selected_trimap_inds.intersection(self.selected_trimap_inds):
                continue
            elif idx in new_selected_trimap_inds:
                trimap = self.load_trimap(self.available_images[idx])
                image_data[DatasetSelection.Trimap] = trimap
            else:
                image_data[DatasetSelection.Trimap] = self.dummy_trimap.clone()
            self.samples[idx] = (img, image_data)
        self.selected_trimap_inds = new_selected_trimap_inds

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __getitem__(self, idx):
        img, sample_data = self.samples[idx]

        sample_data = copy.deepcopy(sample_data)

        def get_crop_safe_zoom_factor(angle_deg):
            angle_rad = math.radians(angle_deg % 180)
            if angle_rad > math.pi / 2:
                angle_rad = math.pi - angle_rad
            return (math.cos(angle_rad) + math.sin(angle_rad))

        if self.training and random.random() > 0.5:
            c, h, w = img.shape

            # rotation

            # -image
            angle = random.uniform(0, 360)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

            # -trimap
            if DatasetSelection.Trimap in sample_data and sample_data[DatasetSelection.Trimap] is not None:
                trimap = sample_data[DatasetSelection.Trimap].unsqueeze(0)
                trimap = TF.rotate(trimap, angle, interpolation=TF.InterpolationMode.NEAREST)
                sample_data[DatasetSelection.Trimap] = trimap

            # -cam
            if DatasetSelection.CAM in sample_data and sample_data[DatasetSelection.CAM] is not None:
                cam = sample_data[DatasetSelection.CAM].unsqueeze(0)
                cam = TF.rotate(cam, angle, interpolation=TF.InterpolationMode.NEAREST)
                sample_data[DatasetSelection.CAM] = cam

            # zoom in based on rotation angle

            zoom_factor = get_crop_safe_zoom_factor(angle)
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            new_h_cam, new_w_cam = int(7 * zoom_factor), int(7 * zoom_factor)

            # -image
            img = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
            img = TF.center_crop(img, [h, w])

            # -trimap
            if DatasetSelection.Trimap in sample_data and sample_data[DatasetSelection.Trimap] is not None:
                trimap = sample_data[DatasetSelection.Trimap]
                trimap = TF.resize(trimap, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
                trimap = TF.center_crop(trimap, [h, w])
                sample_data[DatasetSelection.Trimap] = trimap.squeeze()

            # -cam
            if DatasetSelection.CAM in sample_data and sample_data[DatasetSelection.CAM] is not None:
                cam = sample_data[DatasetSelection.CAM]
                cam = TF.resize(cam, [new_h_cam, new_w_cam], interpolation=TF.InterpolationMode.NEAREST)
                cam = TF.center_crop(cam, [7, 7])
                cam = 0.001 + 0.999 *cam
                sample_data[DatasetSelection.CAM] = cam.squeeze()

            # -bbox
            sample_data[DatasetSelection.BBox] = torch.tensor([-1, -1, -1, -1])

        return img, sample_data

def save_cam_dataset(image_names, cams):
    assert len(image_names) == len(cams)
    os.makedirs(os.path.join(ANNOTATION_DIR, 'heatmaps'), exist_ok=True)
    for fname, cam in zip(image_names, cams):
        torch.save(cam, os.path.join(ANNOTATION_DIR, 'heatmaps', fname))