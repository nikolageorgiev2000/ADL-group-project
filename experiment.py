import copy
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
from enum import Enum
from typing import Dict, List, Tuple, Set
from itertools import product
from datasets import *


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Set up the data directories
data_dir = 'data/images'
annotation_dir = 'data/annotations'  # Add annotations directory

# Get list of image files
image_files = sorted([f for f in os.listdir(data_dir)
                     if f.endswith(('.jpg', '.png'))])[:4]

# Create a figure with 2x4 subplots (original image + trimap for each row)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Load and display first 4 images with their annotations
for idx, img_file in enumerate(image_files):
    # Load original image
    img_path = os.path.join(data_dir, img_file)
    img = Image.open(img_path)

    # Load trimap segmentation
    trimap_file = img_file.replace('.jpg', '.png')
    trimap_path = os.path.join(annotation_dir, 'trimaps', trimap_file)
    trimap = Image.open(trimap_path)

    # Plot in corresponding subplots
    row = idx // 2
    col = idx % 2 * 2  # Multiply by 2 to leave space for annotations

    # Plot original image
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    axes[row, col].set_title(f'{img_file.split("_")[0]} - Original')

    # Plot trimap segmentation
    axes[row, col + 1].imshow(trimap, cmap='tab20')
    axes[row, col + 1].axis('off')
    axes[row, col + 1].set_title('Segmentation Mask')

plt.tight_layout()
plt.show()

# Optional: Display annotation statistics


def print_annotation_info():
    print("\nDataset Annotations Include:")
    print("1. Species/Breed Names (37 categories)")
    print("2. Head Bounding Box (ROI)")
    print("3. Trimap Segmentation:")
    print("   - 1: Pet")
    print("   - 2: Background")
    print("   - 3: Border/Undefined")


print_annotation_info()


def visualize_predictions(model, dataset, num_samples=8, visual_fname='foo.png'):
    model.eval()
    _, axes = plt.subplots(num_samples, 5, figsize=(15, 5*num_samples))

    for idx in range(num_samples):
        img, sample_data = dataset[idx]
        true_mask = sample_data[DatasetSelection.Trimap]
        img = img.cpu()  # Move to CPU
        true_mask = true_mask.cpu()
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred_trimap = torch.argmax(
                pred[DatasetSelection.Trimap], dim=1).squeeze().cpu()
            pred_cam = torch.sigmoid(
                pred[DatasetSelection.CAM]).float().squeeze().cpu()
            pred_bbox = (pred[DatasetSelection.BBox] > 0.0).squeeze().cpu()

        img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor(
            [0.485, 0.456, 0.406])[:, None, None]  # Reverse normalization
        img = img.clip(0, 1)  # Clip to valid range

        # Plot original image
        axes[idx, 0].imshow(img.permute(1, 2, 0))
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')

        # Plot true mask
        axes[idx, 1].imshow(true_mask, cmap='tab20')
        axes[idx, 1].set_title('True Mask')
        axes[idx, 1].axis('off')

        # Plot predicted mask
        axes[idx, 2].imshow(pred_trimap, cmap='tab20')
        axes[idx, 2].set_title('Trimap Mask')
        axes[idx, 2].axis('off')

        # Plot predicted mask
        axes[idx, 3].imshow(pred_cam)
        axes[idx, 3].set_title('Cam Heatmap')
        axes[idx, 3].axis('off')

        # Plot predicted mask
        axes[idx, 4].imshow(pred_bbox, cmap='tab20')
        axes[idx, 4].set_title('Bounding Box')
        axes[idx, 4].axis('off')

    plt.tight_layout()
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(os.path.join('visuals', visual_fname))
    plt.show()


def evaluate_model_metrics(model, dataloader, device):
    """
    Evaluate model performance using various classification metrics.

    Args:
        model: PyTorch model to evaluate
        dataloader: PyTorch DataLoader containing validation/test data
        device: Device to run model on ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing computed metrics
    """
    model.eval()

    # Initialize metric accumulators
    total_accuracy = 0
    total_recall = 0
    total_jaccard = 0
    total_f1 = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm.tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(
                outputs[DatasetSelection.Trimap], dim=1).cpu().numpy()
            # Move targets to CPU
            targets = targets[DatasetSelection.Trimap].cpu().numpy()

            # Calculate metrics for the batch and accumulate
            total_accuracy += accuracy_score(targets.flatten(),
                                             preds.flatten()) * len(targets)
            total_recall += recall_score(targets.flatten(),
                                         preds.flatten(), average='macro') * len(targets)
            total_jaccard += jaccard_score(targets.flatten(),
                                           preds.flatten(), average='macro') * len(targets)
            total_f1 += f1_score(targets.flatten(),
                                 preds.flatten(), average='macro') * len(targets)
            total_samples += len(targets)

    # Calculate average metrics
    metrics = {
        'accuracy': float(total_accuracy / total_samples),
        'recall': float(total_recall / total_samples),
        'jaccard': float(total_jaccard / total_samples),
        'f1': float(total_f1 / total_samples),
    }

    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"Jaccard Index: {metrics['jaccard']:.6f}")
    print(f"F1 Score: {metrics['f1']:.6f}")

    return metrics


def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['best_loss']


class CustomLoss(nn.Module):
    def __init__(self, targets_weights: Dict[DatasetSelection, float]):
        super().__init__()
        self.targets_weights = targets_weights
        self.loss_funcs = {
            DatasetSelection.Trimap: TrimapLoss(),
            DatasetSelection.BBox: BBoxLoss(),
            DatasetSelection.CAM: CamLoss(),
        }
        assert set(self.targets_weights.keys()).issubset(
            self.loss_funcs.keys())

    def forward(self, logits, targets):
        total_loss = None
        for d in set(logits.keys()).intersection(self.targets_weights).intersection(targets):
            l = self.loss_funcs[d](logits[d], targets[d]) * \
                self.targets_weights[d]
            if not total_loss:
                total_loss = l
            else:
                total_loss += l
        return total_loss


class TrimapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100)  # uses mean by default

    def forward(self, logits, targets):
        # print((targets == -100).sum() > 0)
        return torch.nan_to_num(self.loss_fn(logits, targets.long()))


class CamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction='none')  # we'll reduce manually
        self.gap = nn.AvgPool2d(32, 32)

    def forward(self, logits, targets):
        # Mask everything that is not marked with -1 (ignore signal)
        mask = (targets != -1).float()

        resized_logits = self.gap(logits).squeeze()

        loss = self.loss_fn(resized_logits, targets)
        loss = loss * mask  # zero out ignored areas

        # Avoid division by zero
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss.sum() / denom


class BBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, bboxes):
        """
        Args:
            logits: Tensor of shape (B, C=2, H, W) — logits for 2 classes (background/foreground)
            bboxes: Tensor of shape (B, 4) — (xmin, ymin, xmax, ymax) per image; use -1 for missing
        Returns:
            Scalar loss
        """
        B, _, H, W = logits.shape
        background_logits = torch.zeros_like(logits)  # fake bg logits
        logits_2ch = torch.cat([background_logits, logits], dim=1)
        targets = torch.full((B, H, W), -100, dtype=torch.long,
                             device=logits.device)  # Default to ignore

        for i in range(B):
            # print(bboxes[i])
            xmin, ymin, xmax, ymax = bboxes[i]
            if all(bboxes[i] >= 0):  # Valid bbox
                targets[i, :, :] = 0  # Background everywhere
                targets[i, ymin:ymax, xmin:xmax] = 1  # Foreground inside bbox

        loss = self.loss_fn(logits_2ch, targets)
        return loss


def train_model(
    model,
    targets_weights,
    train_dataloader,
    epochs,
    learning_rate,
    optimizer_name='adam',
    scheduler_name='cosine',
    checkpoint_dir='checkpoints',
    resume_from=None
):
    """Train the segmentation model with checkpointing and scheduling"""

    # Set up optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Set up scheduler
    if scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    elif scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    criterion = CustomLoss(targets_weights)
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from:
        model, start_epoch, best_loss = load_checkpoint(model, resume_from)
        print(
            f"Resuming from epoch {start_epoch} with best loss {best_loss:.6f}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nStarting training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for images, image_data in tqdm.tqdm(train_dataloader, disable=True):
            # images = images.to(device)
            # image_data = image_data.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(
                outputs, image_data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        # Update scheduler
        if scheduler_name.lower() == 'cosine':
            scheduler.step()
        else:
            scheduler.step(epoch_loss)

        # Save checkpoint if best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_loss': best_loss
            }
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pth")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_loss': best_loss
            }
            torch.save(
                checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")

    print("\nTraining complete!")
    return model


print("\n=== Using Segmentation Models PyTorch (SMP) for improved performance ===\n")
TARGETS_LIST = [DatasetSelection.CAM,
                DatasetSelection.Trimap, DatasetSelection.BBox]
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
OPTIMIZER_NAME = 'adam'
SCHEDULER_NAME = 'reduce_on_plateau'
CHECKPOINT_DIR = 'checkpoints/'
RESUME_FROM = None  # os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_30.pth')
seed = 42

torch.manual_seed(seed)


class ConvSegHead(nn.Module):
    def __init__(self, out_channels):
        super(ConvSegHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()
        self.feature_extractor = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
        )
        self.feature_extractor.segmentation_head = nn.Identity()

        self.seg_heads = nn.ModuleDict()
        self.seg_heads[DatasetSelection.Trimap.name] = ConvSegHead(
            out_channels=3)
        self.seg_heads[DatasetSelection.CAM.name] = ConvSegHead(out_channels=1)
        self.seg_heads[DatasetSelection.BBox.name] = ConvSegHead(
            out_channels=1)

    def forward(self, img):
        features = self.feature_extractor(img)
        res = {}
        for dataset_name, head in self.seg_heads.items():
            res[DatasetSelection[dataset_name]] = head(features)
        return res


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
    transforms.Lambda(lambda x: x.to(torch.int8)),
])

print('Loading Dataset')
dataset = InMemoryPetSegmentationDataset(
    DATA_DIR, ANNOTATION_DIR, targets_list=TARGETS_LIST)
# dataset_perm = torch.randperm(len(dataset))

GT_PROPORTIONS = [0.005, 0.01, 0.05]
LOSS_WEIGHTS = [0.0, 0.1, 0.5]


for idx, experiment_weights in enumerate(product(GT_PROPORTIONS, LOSS_WEIGHTS, LOSS_WEIGHTS)):
    gt_prop, cam_loss_weight, bbox_loss_weight = experiment_weights
    print(gt_prop, cam_loss_weight, bbox_loss_weight)
    # set based on how much of the dataset we can use
    dataset.change_gt_proportion(gt_prop)

    # Create train/val split
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = torch.utils.data.Subset(
        dataset, range(train_size))

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create SMP model - Using UNet with ResNet34 encoder pre-trained on ImageNet
    smp_model = CustomUNet().to(device)

    model = train_model(
        smp_model,
        {DatasetSelection.Trimap: 1.0, DatasetSelection.CAM: cam_loss_weight,
            DatasetSelection.BBox: bbox_loss_weight},
        train_dataloader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer_name=OPTIMIZER_NAME,
        scheduler_name=SCHEDULER_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        resume_from=RESUME_FROM
    )
    smp_model, smp_epoch, smp_best_loss = load_checkpoint(
        smp_model, 'checkpoints/best_model.pth')
    print(smp_epoch)

    # reset GT proportion to perform evaluation on trimaps correctly
    dataset.change_gt_proportion(1.0)
    val_dataset = torch.utils.data.Subset(
        dataset, range(train_size, train_size + val_size))
    val_dataloader = DataLoader(
        val_dataset, batch_size=4*BATCH_SIZE, shuffle=False)
    metrics = visualize_predictions(
        smp_model, val_dataset, visual_fname=f"{idx}.jpg")
    metrics = evaluate_model_metrics(model, val_dataloader, device)
    os.makedirs('run_results', exist_ok=True)
    with open(f'run_results/{idx}.txt', 'w') as file:
        res_str = str(experiment_weights)+'\n'+str(metrics)
        file.write(res_str)

    del train_dataset, val_dataset, train_dataloader, val_dataloader
