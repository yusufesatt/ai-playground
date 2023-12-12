# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 01:07:25 2023

@author: yusuf
"""
# %%
import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset

# %%

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
        self.fc = nn.Linear(64 * 61 * 61, 10)
        self.relu_fc = nn.ReLU()
        
    def forward(self, x):
        
        x = x.cuda()
        
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
    
# %%
path = r"C:\Users\yusuf\Desktop\New Plant Diseases Dataset(Augmented)\train"

# %%


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}  # İndeksleri sınıf isimleri ile eşleştirme
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

dataset = CustomDataset(root_dir=path, transform=transform)
class_names = dataset.classes

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# %%

model = CNNModel().to(device)
model.load_state_dict(torch.load(r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.3/5-epoch-lr-001.pth"))
model.eval()  

image_path = r"C:\Users\yusuf\Desktop\New Plant Diseases Dataset(Augmented)\valid\Tomato___Bacterial_spot\0ab54691-ba9f-4c1f-a69b-ec0501df4401___GCREC_Bact.Sp 3170.JPG"  # Tahmin edilecek resmin dosya yolu
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
input_image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    model_output = model(input_image)

_, predicted_class = torch.max(model_output, 1)
predicted_class = predicted_class.item()

predicted_class_name = class_names[predicted_class]

print(f'Tahmin edilen sınıf: {predicted_class_name}')
