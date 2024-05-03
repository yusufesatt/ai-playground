# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:46:13 2024

@author: yusuf
"""

# %%
# VGG13 Model

# %%
# Import Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torchsummary import summary
import glob
from tqdm import tqdm

# %%

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        self.data = self.load_data()
        
    def load_data(self):
        data = []
        
        for class_name in self.classes:
            class_folder = os.path.join(self.root_dir, class_name)
            files = os.listdir(class_folder)
            for file in files:
                data.append((os.path.join(class_folder, file), self.classes.index(class_name)))
        
        return data

    
    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
# %%

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# %%
# Dataset prepare

batch_size = 32

train_path = r"C:\Users\yusuf\Documents\AI\Datasets\New Plant Diseases Dataset(Augmented)\train"
test_path = r"C:\Users\yusuf\Documents\AI\Datasets\New Plant Diseases Dataset(Augmented)\valid"

train_dataset = CustomDataset(root_dir=train_path, transform=transform)
test_dataset = CustomDataset(root_dir=test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


for images, labels in train_loader:
    break

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Total number of classes: {num_classes}")
print(f"Classes: {class_names}")

# %%
# Device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %%
# Create Model

class VGG13(nn.Module):
    def __init__(self, in_channels=3):
        super(VGG13, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # Converts the dimension of the input tensor to a specified target dimension
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = VGG13().to(device)
summary(model, input_size=(3, 64, 64))


# %%

num_epochs = 5

# Cross Entropy Loss
loss_func = nn.CrossEntropyLoss()

# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# %%

train_count = len(glob.glob(train_path+'/*/*'))
test_count = len(glob.glob(test_path+'/*/*'))

print(f"Total number of images in Training dataset: {train_count}, Test dataset: {test_count}")

# %%

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_accuracy = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    
    # Evaluation and training on training dataset
    
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    
    for i, (images,labels) in enumerate(tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch')):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = loss_func(outputs,labels)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.cpu().data*images.size(0)
        _,prediction = torch.max(outputs.data,1)
        
        train_accuracy += int(torch.sum(prediction==labels.data))
        
    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    
    # Evaluation on testing dataset
    
    model.eval()
    
    test_loss_total = 0.0
    test_accuracy = 0.0
    
    for i, (images,labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        _,prediction = torch.max(outputs.data,1)
        test_accuracy += int(torch.sum(prediction==labels.data))
        test_loss_total += loss.item()
        
    test_accuracy = test_accuracy / test_count
    test_loss = test_loss_total / len(test_loader)
    
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
    
    print(
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')

    
    #Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint.model')
        best_accuracy = test_accuracy
        best_epoch = epoch
    else:
        pass

print(f"Training operation completed. The model with {best_accuracy:.2%} value was recorded and {best_epoch+1}")
            

# %%
# Test structure vgg13
"""
from torchvision.models import vgg13
model = vgg13(pretrained=True)
model.to(device)
summary(model, input_size=(3, 256, 256))
"""
# %%
