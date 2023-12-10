from __future__ import annotations

import abc
import os
from abc import abstractmethod
from dataclasses import dataclass

import torch
from torchvision import transforms

from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer
from iris_recognition.trainset import Trainset
from iris_recognition.transforms.horizontal_stack import HorizontalStack


@dataclass
class TrainingParams:
    """
    Class containing all settable training parameters
    """
    num_epochs: int
    learning_rate: float
    weight_decay: float


class PretrainedModel(abc.ABC):
    """
    Class representing pretrained model. Used to finetune model
    """

    def __init__(self, prefix: str | None = None) -> None:
        self.path_organizer = PathOrganizer(prefix)
        self.logger = get_logger("Pretrained Model " + self.name)
        self.model: torch.nn.Module | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: name of the model
        """

    @abstractmethod
    def prepare_classification_layers(self, num_classes: int) -> None:
        """
        Adds classification layers to the model
        :param num_classes: number of classes
        """

    def train(self, trainset: Trainset, params: TrainingParams) -> None:
        """
        Train model
        """
        if self.model is None:
            raise ValueError("Model is not initialized properly")

        self.logger.info("Preparing classification layers")
        self.prepare_classification_layers(trainset.num_classes())
        self.logger.info("Done preparing classification layers")

        self.logger.info(f"Starting training model {self.name}")
        self.logger.info(f"Training set len: {trainset.train_len()}, Validation set len: {trainset.valid_len()}")

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
        """
        Saves model to use in the testing
        """
        model_filename = "model.pt"
        dir_to_save = self.path_organizer.get_finetuned_model_dir(self.name, tag)
        self.logger.info(f"Saving model to dir {dir_to_save} to file {model_filename}")
        os.makedirs(dir_to_save, exist_ok=True)
        torch.save(self.model, os.path.join(dir_to_save, model_filename))

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            HorizontalStack(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
