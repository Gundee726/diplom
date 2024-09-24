import torch
from torch.utils.data import DataLoader
import pickle
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

# Load the preprocessed datasets
with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

with open('val_dataset.pkl', 'rb') as f:
    val_dataset = pickle.load(f)

# Access the original dataset to get the number of classes
original_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the ResNet-50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Get the number of classes
num_classes = len(original_dataset.dataset.classes)  # Access the classes from the original dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Optimize the model

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'fashion_model.pth')

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")