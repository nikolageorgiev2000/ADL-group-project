# %%
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# %%

import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

# Set up the data directories
data_dir = '.data/images'
annotation_dir = '.data/annotations'  # Add annotations directory

# Get list of image files
image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])[:4]

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

# %%
# Set up dataset and dataloader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PetSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_dir, transform):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Load trimap
        trimap_file = self.image_files[idx].replace('.jpg', '.png')
        trimap_path = os.path.join(self.annotation_dir, 'trimaps', trimap_file)
        trimap = Image.open(trimap_path)
        
        # Apply transforms
        img = self.transform(img)
        trimap = torch.tensor(np.array(trimap.resize((256, 256))), dtype=torch.long)
        
        return img, trimap


# %%

# Define the segmentation model
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final = nn.Conv2d(32, 3, 1)  # 3 classes: pet, background, border
        
    def forward(self, x):
        # Encoding
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(self.conv3(x2))
        
        # Decoding
        x = F.relu(self.upconv1(x3))
        x = F.relu(self.upconv2(x))
        return self.final(x)


model = SimpleSegNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(train_dataloader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# %%

# Visualization function
def visualize_predictions(model, dataset, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for idx in range(num_samples):
        img, true_mask = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()
        
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
    from sklearn.metrics import accuracy_score, recall_score, jaccard_score, f1_score
    import numpy as np
    
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.numpy().flatten())

    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds, average='macro'),
        'jaccard': jaccard_score(all_targets, all_preds, average='macro'),
        'f1': f1_score(all_targets, all_preds, average='macro')
    }

    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Jaccard Index: {metrics['jaccard']:.4f}") 
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    return metrics

# %% 

# %%

# preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    # transforms.Lambda(preprocess_input),
])
dataset = PetSegmentationDataset(data_dir, annotation_dir, transform)

# %%
def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epoch'], checkpoint['best_loss']

def train_model(
    model,
    dataset,
    train_samples=128,
    batch_size=64,
    epochs=100,
    learning_rate=3e-5,
    optimizer_name='adam',
    scheduler_name='cosine',
    checkpoint_dir='checkpoints',
    resume_from=None
):
    """Train the segmentation model with checkpointing and scheduling"""
    
    # Create train/val split
    train_size = train_samples
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Set up scheduler
    if scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from:
        model, start_epoch, best_loss = load_checkpoint(model, resume_from)
        print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\nStarting training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
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
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")

    print("\nTraining complete!")
    return model


# %%
print("\n=== Using Segmentation Models PyTorch (SMP) for improved performance ===\n")

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Create SMP model - Using UNet with ResNet34 encoder pre-trained on ImageNet
smp_model = smp.Unet(
    encoder_name="resnet34",        # Choose encoder, e.g. resnet34
    encoder_weights="imagenet",     # Use pre-trained weights
    in_channels=3,                  # Number of input channels (RGB)
    classes=3,                      # Number of output classes (pet, background, border)
).to(device)

train_model(smp_model, dataset, train_samples=128, batch_size=64, epochs=100, learning_rate=3e-5, optimizer_name='adam', scheduler_name='cosine', checkpoint_dir='checkpoints', resume_from=None)

ckpt_model, ckpt_epoch, ckpt_best_loss = load_checkpoint(smp_model, 'checkpoints/best_model.pth')

metrics = visualize_predictions(smp_model, (train_dataset))
# %%
