import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.image_paths, self.labels = self.load_data_U(base_dir)
        self.transform = transform

    def load_data_U(self, base_dir):
        image_paths = []
        labels = []
        label_map = {}
        current_label = 0

        images = os.listdir(base_dir)
        for image in images:
            if image.endswith('.tiff'):
                parts = image.split('_')
                person = parts[0]
                side = parts[1]

                label_key = f'{person}_{side}'
                if label_key not in label_map:
                    label_map[label_key] = current_label
                    current_label += 1

                image_path = os.path.join(base_dir, image)
                image_paths.append(image_path)
                labels.append(label_map[label_key])

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
base_dir = 'CLASSES_400_300_Part2'
dataset = MyDataset(base_dir, transform)

# Splitting dataset into train, validation, and test sets
valid_size = 0.3
test_size = 0.5
batch_size = 1
num_workers = 0

indices = list(range(len(dataset)))
np.random.shuffle(indices)
split = int(np.floor(valid_size * len(dataset)))
train_idx, valid_idx = indices[split:], indices[:split]

valid_idx = valid_idx[int(np.floor(test_size * len(valid_idx))):]
test_idx = valid_idx[:int(np.floor(test_size * len(valid_idx)))]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# DataLoaders
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg16(weights='DEFAULT')
num_classes = len(set(dataset.labels))  # Update num_classes based on dataset
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
model.to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

print(1)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_sampler)
        train_acc = running_corrects.double() / len(train_sampler)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_loader)
        val_acc = running_corrects.double() / len(val_loader)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'.format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))


# Training the model
train(model, train_loader, valid_loader, criterion, optimizer, 100)

# Save the model
# torch.save(model.state_dict(), 'model_state_dict.pth')


torch.save(model, 'vgg16.pth')