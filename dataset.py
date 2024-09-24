import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset
import pickle

# Define transformations: Resize, convert to Tensor, and Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_dir = '/Users/gundalaib/Downloads/img'
# Load the dataset
original_dataset = datasets.ImageFolder(img_dir, transform=transform)

# Limit to 1000 images for testing
limited_dataset = Subset(original_dataset, range(0, min(1000, len(original_dataset))))

# Split dataset into training and validation (80% train, 20% validation)
train_size = int(0.8 * len(limited_dataset))
val_size = len(limited_dataset) - train_size
train_dataset, val_dataset = random_split(limited_dataset, [train_size, val_size])

# Save the datasets to files
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

with open('val_dataset.pkl', 'wb') as f:
    pickle.dump(val_dataset, f)