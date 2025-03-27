import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

# CAM Model using ResNet backbone
class CAMModel(nn.Module):
    def __init__(self, num_classes=37):  # Oxford-IIIT Pet has 37 categories
        super(CAMModel, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        self.fc = nn.Linear(2048, num_classes)  # 2048 is output channels for ResNet-50
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        
        # Save features for CAM generation
        self.last_features = features
        
        # Global Average Pooling
        x = self.gap(features)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x
    
    def predict(self, image):
        """
        Takes a PIL image, applies transformations and returns prediction
        
        Args:
            image: PIL Image
            
        Returns:
            model output
        """
        device = next(self.parameters()).device
        self.eval()
        
        # Apply transforms
        img_tensor = self.transforms(image)
        
        # Forward pass
        with torch.no_grad():
            output = self(img_tensor.unsqueeze(0).to(device))
            
        return output, img_tensor

    def generate_cam(self, img, label=None):
        """
        Generate Class Activation Map for an image.
        
        Args:
            model: The trained model with CAM capability
            img_tensor: Preprocessed image tensor
            label: Class label for which to generate CAM. If None, the predicted class is used.
        
        Returns:
            CAM numpy array
        """
        model = self
        device = next(model.parameters()).device
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
        
        # get class prediction as well
        if label is None:
            _, pred_label = torch.max(output, 1)
            pred_label = pred_label.item()
        else:
            pred_label = label
                    
        # Get weights from the final FC layer for the specified class
        fc_weights = model.fc.weight[pred_label].cpu()
        
        # Get feature maps from the last convolutional layer
        feature_maps = model.last_features.squeeze(0).cpu()
        
        # Calculate weighted sum of feature maps
        cam = torch.zeros(feature_maps.shape[1:])
        for i, weight in enumerate(fc_weights):
            cam += weight * feature_maps[i]
        
        # Apply ReLU and normalize
        cam = torch.maximum(cam, torch.tensor(0.0))
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy and resize
        cam = cam.detach().numpy()
        
        return pred_label, cam
    

    def plot_cam_overlay(self, img, cam, label=None, sampled_points=None, save_path=None):
        """
        Plot image with CAM overlay.
        
        Args:
            img: Original image as numpy array (H,W,C) with RGB channels
            cam: Resized CAM numpy array (same H,W as img)
            label: Optional label to display in title
        """
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay (60% original image, 40% heatmap)
        overlay = np.uint8(0.6 * img + 0.4 * heatmap)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        if label is not None:
            plt.title(f'Class: {label}')
            
        # plot points
        if sampled_points is not None:
            for idx in sampled_points:
                plt.scatter(idx[0], idx[1], c='red', s=10)
                
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
    def sample_cam_points(self, cam_resized, strategy='local_peak'):
        assert strategy in ['peak', 'local_peak', 'random'], "Invalid strategy"
        if strategy == 'peak':
            peak = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
            return np.array([[peak[1], peak[0]]])
        
        elif strategy == 'local_peak':
            # find local peaks
            kernel = np.ones((3,3))
            cam_dilated = cv2.dilate(cam_resized, kernel)
            local_peaks = np.where(cam_resized == cam_dilated)
            local_peaks = np.stack(local_peaks, axis=1)
            return local_peaks
            
        else:
            cam_resized = np.where(cam_resized < 0.7, 0, cam_resized)    
            heatmap_distribution = (cam_resized.flatten()/cam_resized.sum())
            heatmap_distribution = heatmap_distribution**2
            sampled_points = np.random.choice(np.arange(cam_resized.size), size=50, p=cam_resized.flatten()/cam_resized.sum())
            sampled_points = np.unravel_index(sampled_points, cam_resized.shape)
            sampled_points = np.stack(sampled_points, axis=1)
            # reverse x, y
            sampled_points = sampled_points[:, ::-1]
            return sampled_points
        # Add this method to the CAMModel class

    def batch_generate_cam(self, batch_tensors, batch_labels):
        """Generate CAMs for a batch of images"""
        self.eval()
        device = next(self.parameters()).device
        batch_size = batch_tensors.size(0)
        batch_cams = []
        batch_labels_processed = []
        
        with torch.no_grad():
            # Forward pass for the whole batch
            outputs = self(batch_tensors.to(device))
            
            # Get feature maps for the whole batch
            batch_features = self.last_features.cpu()
            
            # Process each image in the batch
            for i in range(batch_size):
                label = batch_labels[i]
                # Get weights from the final FC layer for the specified class
                fc_weights = self.fc.weight[label].cpu()
                
                # Get feature maps for this image
                feature_maps = batch_features[i]
                
                # Calculate weighted sum of feature maps
                cam = torch.zeros(feature_maps.shape[1:])
                for j, weight in enumerate(fc_weights):
                    cam += weight * feature_maps[j]
                
                # Apply ReLU and normalize
                cam = torch.maximum(cam, torch.tensor(0.0))
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                # Convert to numpy
                cam = cam.detach().numpy()
                batch_cams.append(cam)
                batch_labels_processed.append(label)
        
        return batch_labels_processed, batch_cams

if __name__ == "__main__":
    from utils.logger import setup_folder_and_logger
    experiment_folder, logger = setup_folder_and_logger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset without transforms
    test_dataset = OxfordIIITPet(root='./data', 
                            split='test',
                            target_types="category",
                            transform=None,  # No transforms here
                            download=True)
    
    # make new CAMModel and load the pre-trained weights
    model = CAMModel()
    model.load_state_dict(torch.load('./weakly_supervised/models/cam_model.pth', map_location=device, weights_only=False))
    model = model.to(device)
    logger.info("Loading model:\n", model) 
    
    # idx = 0 # Change this index to visualize different images
    
    # # Get image and label from dataset - now the image is a PIL image
    # original_img, label = test_dataset[idx]
    
    # # Apply transforms inside the predict method
    # output, img_tensor = model.predict(original_img)
    
    # logger.info(f'Class: {label}')
    
    # # Generate CAM
    # pred_label, pred_cam = model.generate_cam(img_tensor, label)
    
    # # Convert original image to numpy array for plotting
    # original_np = np.array(original_img)

    # # Resize CAM to match original image size
    # cam_resized = cv2.resize(pred_cam, (original_img.width, original_img.height))
    
    # points = model.sample_cam_points(cam_resized)

    # # Plot overlay
    # model.plot_cam_overlay(original_np, cam_resized, pred_label, sampled_points=points, save_path=f'{experiment_folder}/cam_overlay.png')









    # Define batch size
    BATCH_SIZE = 4

    # Randomly sample 10 samples
    num_samples = 10
    
    num_batches = num_samples // BATCH_SIZE
    
    class_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    class_indices = np.sort(class_indices)

    def pil_to_tensor(pil_image):
        transform = transforms.Compose([
            transforms.ToTensor(),        
            transforms.Resize((224, 224)),
        ])
        return transform(pil_image)

    # Process in batches
    for batch_start in tqdm(range(0, num_batches, BATCH_SIZE)):
        batch_indices = class_indices[batch_start:batch_start+BATCH_SIZE]
        logger.info(f"Processing batch: {batch_indices}")
        # Prepare batch data
        batch_images = []
        batch_labels = []
        batch_original_images = []
        batch_classes = []
        
        for idx in batch_indices:
            image, label = test_dataset[idx]
            batch_original_images.append(image)
            batch_images.append(pil_to_tensor(image))
            batch_labels.append(label)
            batch_classes.append(test_dataset.classes[label].replace(" ", "_").lower())
        
        logger.info(f"Batch classes: {batch_classes}")
        logger.info(f"Batch labels: {batch_labels}")
        logger.info(f"Batch images: {len(batch_images)}")
        
        # Stack preprocessed images for batch processing
        batch_tensors = torch.stack(batch_images)
        logger.info(f"Stacked batch tensor shape: {batch_tensors.shape}")
        
        # Generate CAMs in batch
        batch_pred_labels, batch_cams = model.batch_generate_cam(batch_tensors, batch_labels)
        
        logger.info(f"Batch predicted labels: {batch_pred_labels}")
        
        # Process each image with its CAM
        for i in range(len(batch_indices)):
            logger.info(f"Plotting each CAM {i}")
            image = batch_original_images[i]
            label = batch_labels[i]
            class_name = batch_classes[i]
            cam = batch_cams[i]
            
            idx_in_dataset = batch_indices[i]
            output_idx = batch_start + i
            
            # Resize CAM to match image dimensions
            cam_resized = cv2.resize(cam, (image.width, image.height))
            
            # Sample points from CAM
            points = model.sample_cam_points(cam_resized, strategy='peak')
            
            # Plot CAM overlay
            model.plot_cam_overlay(
                np.array(image), 
                cam_resized, 
                label, 
                sampled_points=points, 
                save_path=f'{experiment_folder}/cam_overlay_{output_idx}.png'
            )
            