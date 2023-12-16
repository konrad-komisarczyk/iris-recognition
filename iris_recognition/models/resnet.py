from __future__ import annotations

from torchvision.models import resnet152
import torch

from iris_recognition.models.model import Model


class Resnet(Model):
    """
    Pretrained resnet152 model class
    """

    @property
    def name(self) -> str:
        return "resnet152"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = resnet152(weights='DEFAULT')

        # Freeze all the layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with additional layers
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)  # Final layer for 450 classes
        )
        
        self.logger.debug("Done loading model and preparing classification layer")
