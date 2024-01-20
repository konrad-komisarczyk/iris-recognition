from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet152
import os
import numpy as np
import torch
from torchvision.utils import _log_api_usage_once
from PIL import Image


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
data_dir = os.path.join(base_dir, "ubiris2_1_preprocessed")


class HorizontalStack(torch.nn.Module):
    """Horizontally stack the given image two times.
    """

    def __init__(self, p=2):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        imgs = [img for _ in range(self.p)]

        imgs_comb = np.vstack(imgs)
        imgs_comb = Image.fromarray(imgs_comb)

        return imgs_comb

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


transform = transforms.Compose([
    HorizontalStack(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = datasets.ImageFolder(data_dir, transform=transform)

valid_size = 0.3
test_size = 0.5
batch_size = 1
num_workers = 0

indices = list(range(len(data)))
np.random.shuffle(indices)
split = int(np.floor(valid_size * len(data)))
train_idx, valid_idx = indices[split:], indices[:split]

valid_idx = valid_idx[int(np.floor(test_size * len(valid_idx))):]
test_idx = valid_idx[:int(np.floor(test_size * len(valid_idx)))]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

# # Code to visualize the distribution of samples for each class in train, test and validation datasets
# import pandas as pd
# import seaborn as sns
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt
#
# idx2class = {v: k for k, v in data.class_to_idx.items()}
#
#
# def get_class_distribution_loaders(dataloader_obj, dataset_obj):
#     count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
#
#     for _, j in dataloader_obj:
#         y_idx = j.item()
#         y_lbl = idx2class[y_idx]
#         count_dict[str(y_lbl)] += 1
#
#     return count_dict
#
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
# sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution_loaders(train_loader, data)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Train Set')
# sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution_loaders(valid_loader, data)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Val Set')
# plt.show()

model = resnet152(weights='DEFAULT')  # pretained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(os.listdir(data_dir))
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


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
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))


train(model, train_loader, valid_loader, criterion, optimizer, 100)

