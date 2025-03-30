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

def visualize_predictions(model, dataset, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    for idx in range(num_samples):
        img, sample_data = dataset[idx]
        true_mask = sample_data[DatasetSelection.Trimap]
        img = img.cpu()  # Move to CPU
        true_mask = true_mask.cpu()
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()

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
        axes[idx, 2].imshow(pred_mask, cmap='tab20')
        axes[idx, 2].set_title('Predicted Mask')
        axes[idx, 2].axis('off')

    plt.tight_layout()
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
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = targets.cpu().numpy()  # Move targets to CPU

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
        'accuracy': total_accuracy / total_samples,
        'recall': total_recall / total_samples,
        'jaccard': total_jaccard / total_samples,
        'f1': total_f1 / total_samples
    }

    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Jaccard Index: {metrics['jaccard']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics


def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['best_loss']


def train_model(
    model,
    train_dataloader,
    epochs=20,
    learning_rate=3e-5,
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

    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from:
        model, start_epoch, best_loss = load_checkpoint(model, resume_from)
        print(
            f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nStarting training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm.tqdm(train_dataloader):
            # images = images.to(device)
            # targets = targets.to(device)

            optimizer.zero_grad()
            # print(images.shape, targets.shape)
            outputs = model(images)
            # print(outputs.shape, targets.shape)
            loss = criterion(
                outputs, targets[DatasetSelection.Trimap].to(torch.long))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

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
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
OPTIMIZER_NAME = 'adam'
SCHEDULER_NAME = 'reduce_on_plateau'
CHECKPOINT_DIR = 'checkpoints/'
RESUME_FROM = None #os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_30.pth')
seed = 42

torch.manual_seed(seed)

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

train_dataset = InMemoryPetSegmentationDataset(
    DATA_DIR, ANNOTATION_DIR, targets_list=[DatasetSelection.Trimap])


# Create train/val split
train_size = int(0.8*len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size])

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create SMP model - Using UNet with ResNet34 encoder pre-trained on ImageNet
smp_model = smp.Unet(
    encoder_name="resnet34",        # Choose encoder, e.g. resnet34
    encoder_weights="imagenet",     # Use pre-trained weights
    in_channels=3,                  # Number of input channels (RGB)
    # Number of output classes (pet, background, border)
    classes=3,
).to(device)

model = train_model(
    smp_model,
    train_dataloader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    optimizer_name=OPTIMIZER_NAME,
    scheduler_name=SCHEDULER_NAME,
    checkpoint_dir=CHECKPOINT_DIR,
    resume_from=RESUME_FROM
)

# smp_model(next(iter(train_dataloader))[0])

# # smp_model, smp_epoch, smp_best_loss = load_checkpoint(smp_model, '/content/drive/My Drive/PetSegmentation/checkpoints/best_model.pth')

metrics = visualize_predictions(smp_model, (val_dataset))
