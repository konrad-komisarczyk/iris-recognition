from __future__ import annotations

import abc
import os
import pathlib
from abc import abstractmethod
from dataclasses import dataclass

import torch
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from PIL.Image import Image

from iris_recognition.extracted_features import ExtractedFeatures
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
    batch_size: int


class Model(abc.ABC):
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
    def prepare_pretrained(self, num_classes: int) -> None:
        """
        Loads pretrained model and adds classification layer
        :param num_classes: number of classes
        """

    def _train_epoch(self, device, train_loader, optimizer, criterion) -> tuple[float, float]:
        # Set the model to train mode
        self.model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0
        n_batches = 0

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
            n_batches += 1

        # Calculate the train loss and accuracy
        train_loss = running_loss / n_batches
        train_acc = running_corrects / n_batches

        return train_loss, train_acc

    def _eval_epoch(self, device, val_loader, criterion) -> tuple[float, float]:
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0
        num_batches = 0

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
                num_batches += 1

        # Calculate the validation loss and accuracy
        val_loss = running_loss / num_batches
        val_acc = running_corrects / num_batches
        return val_loss, val_acc

    def train(self, trainset: Trainset, valset: Trainset | None, params: TrainingParams) -> None:
        """
        Train model
        :param trainset: Trainset object for training set
        :param valset: optional Trainset object for validation set
        :param params: TrainingParams object
        """
        if self.model is None:
            raise ValueError("Model is not initialized properly")

        self.logger.info(f"Starting training model {self.name}")
        self.logger.info(f"Training set len: {len(trainset)}, Validation set len: {len(valset) if valset else 0}")
        train_loader = trainset.get_dataloader(batch_size=params.batch_size)
        val_loader = valset.get_dataloader(batch_size=params.batch_size) if valset else None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Device that will be used: {device}")
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

        for epoch in range(params.num_epochs):
            train_loss, train_acc = self._train_epoch(device, train_loader, optimizer, criterion)
            train_metrics_str = f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}"

            val_metrics_str = ""
            if valset:
                val_loss, val_acc = self._eval_epoch(device, val_loader, criterion)
                val_metrics_str = f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"

            # Print the epoch results
            self.logger.info(f'Epoch [{epoch + 1}/{params.num_epochs}]: {train_metrics_str}; {val_metrics_str}')

    @staticmethod
    def get_transform() -> transforms.Compose:
        """
        :return: transform that should be applied to image inputs of the model
        """
        return transforms.Compose([
            HorizontalStack(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def save(self, tag: str) -> None:
        """
        Saves finetuned model to use in the testing
        :param tag: training tag
        """
        model_path = self.path_organizer.get_finetuned_model_path(self.name, tag)
        os.makedirs(pathlib.Path(model_path).parent, exist_ok=True)
        self.logger.info(f"Saving model to path {model_path}")
        torch.save(self.model, model_path)

    def load_finetuned(self, tag: str) -> None:
        """
        Loads finetuned model
        :param tag: training tag
        """
        model_path = self.path_organizer.get_finetuned_model_path(self.name, tag)
        self.model = torch.load(model_path)

    def log_node_names(self) -> None:
        train_nodes, eval_nodes = get_graph_node_names(self.model)
        self.logger.info("Model node names that can be used for feature extraction:")
        self.logger.info(f"Train nodes: {train_nodes}")
        self.logger.info(f"Eval nodes: {eval_nodes}")

    def extract_features(self, node_name: str, preprocessed_image: Image) -> ExtractedFeatures:
        """
        :param node_name: name of the node to extract features from
        :param preprocessed_image: preprocessed image
        :return: extracted features
        """
        feature_extractor = create_feature_extractor(self.model, [node_name])
        transform = self.get_transform()
        transformed_image = transform(preprocessed_image)
        extracted_features_value = feature_extractor(transformed_image)[node_name]
        return ExtractedFeatures(extracted_features_value)