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

from datasets import *


# CAM Model using ResNet backbone
class CAMModel(nn.Module):
    def __init__(self, num_classes=37):  # Oxford-IIIT Pet has 37 categories
        super(CAMModel, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification layer
        self.fc = nn.Linear(2048, num_classes)  # 2048 is output channels for ResNet-50
    
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

def train_model(model, train_loader, num_epochs=5):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Training on {device}...")

    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, image_data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = image_data[DatasetSelection.Class].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

    # Save the model checkpoint
    torch.save(model.state_dict(), './models/cam_model.pth')
    return model

print("Creating model...")
model = CAMModel()
if not os.path.exists('./models/'): os.mkdir('./models')
# Set to True if you want to train, False to load pre-trained
TRAIN_MODEL = False

if TRAIN_MODEL:
    train_dataset = InMemoryPetSegmentationDataset(
        DATA_DIR, ANNOTATION_DIR, targets_list=[DatasetSelection.Class])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = train_model(model, train_loader, num_epochs=10)
else:
    # Load pretrained model if it exists
    if os.path.exists('./models/cam_model.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load('./models/cam_model.pth', map_location=device, weights_only=False))
        model = model.to(device)
        print("Loaded pre-trained model")
    else:
        print("No pre-trained model found")