# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:39:44 2023

@author: yusuf
"""

# %%
# Import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torchsummary import summary

# %%
# Dataset prepare

path = r"C:\Users\yusuf\Desktop\New Plant Diseases Dataset(Augmented)\train"

# %%


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)} # Matching indexes with class names
        self.images = self.get_images()

    def get_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    

# %%

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# %%

dataset = CustomDataset(root_dir=path, transform=transform)
class_names = dataset.classes

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
test_path = r"C:\Users\yusuf\Desktop\New Plant Diseases Dataset(Augmented)\valid"

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomDataset(root_dir=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# %%
# Create CNN Model

num_classes = 10

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Conv1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        
        # Conv2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # FC layers
        self.fc = nn.Linear(64 * 61 * 61, 10) # Learn the values from maxpool on the model summary
        self.relu_fc = nn.ReLU()
        
    def forward(self, x):
        
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        
        
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        
        # Flattening
        x = self.flatten(x)
    
        # FC
        x = self.fc(x)
        x = self.relu_fc(x)
        
        return x
    
batch_size = 32
num_epochs = 10
n_iters = (len(dataset) / batch_size) * num_epochs

# Create CNN
model = CNNModel().to(device)
summary(model, input_size=(3, 256, 256))

# Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%  
def calculate_metrics(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to the gpu
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(dataloader)

    model.train()  # Put the model back into training mode

    return average_loss, accuracy

# %%
train_losses = []
train_accuracies = []

test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        # Convert data to tensors
        images, labels = images.to(device), labels.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate Loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient update
        optimizer.step()

    #  Calculate loss and accuracy
    train_loss, train_accuracy = calculate_metrics(model, train_loader, criterion)
    test_loss, test_accuracy = calculate_metrics(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')
        
# %%

# Model save
# torch.save(model.state_dict(), '5-epoch-lr-001.pth')
# torch.save(optimizer.state_dict(), '5-epoch-lr-001-optimizer.pth')




