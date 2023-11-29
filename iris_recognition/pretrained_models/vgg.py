from __future__ import annotations

from typing import Any

import os
import torch
from torchvision import transforms
from torchvision.models import vgg16

from iris_recognition.pretrained_models import PretrainedModel
from iris_recognition.pretrained_models.pretrained_model import TrainingParams
from iris_recognition.trainset import Trainset


class VggPretrained(PretrainedModel):
    """
    Pretrained VGG16 model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)

        self.logger.debug("Loading model")
        self.model = vgg16(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "VGG16"

    def train(self, trainset: Trainset, params: TrainingParams) -> None:
        self.logger.info(f"Starting training model {self.name}")
        self.logger.info(f"Training set len: {trainset.train_len()}, Validation set len: {trainset.valid_len()}")

        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, trainset.num_classes())

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device that will be used: {device}")

        self.model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

        train_loader, val_loader = trainset.get_dataloaders()

        for epoch in range(params.num_epochs):
            # Set the model to train mode
            self.model.train()

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
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate the train loss and accuracy
            train_loss = running_loss / trainset.train_len()
            train_acc = running_corrects.double() / trainset.train_len()

            # Set the model to evaluation mode
            self.model.eval()

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
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Update the running loss and accuracy
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            # Calculate the validation loss and accuracy
            val_loss = running_loss / len(val_loader)
            val_acc = running_corrects.double() / len(val_loader)

            # Print the epoch results
            self.logger.info(f'Epoch [{epoch + 1}/{params.num_epochs}], train loss: {train_loss:.4f}, '
                             f'train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

    def save_as_finetuned(self, tag: str) -> None:
        # TODO: does this delete classification layer?
        model_filename = 'vgg16.pth'
        dir_to_save = self.path_organizer.get_finetuned_model_dir(self.name, tag)
        self.logger.info(f"Saving model to dir {dir_to_save} to file {model_filename}")
        os.makedirs(dir_to_save, exist_ok=True)
        torch.save(self.model, os.path.join(dir_to_save, model_filename))

    def get_transform(self) -> Any:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
