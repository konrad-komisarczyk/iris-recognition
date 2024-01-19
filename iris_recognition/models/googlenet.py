from __future__ import annotations

from torch import nn
from torchvision.models import googlenet

from iris_recognition.models.model import Model


class Googlenet(Model):
    """
    Pretrained GoogLeNet model class
    """

    @property
    def name(self) -> str:
        return "GoogLeNet"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = googlenet(weights='DEFAULT')
        num_ftrs = self.model.fc.in_features
        # Replace the final fully connected layer (fc) with the new layers
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Final layer for num_classes classes
        )
        self.logger.debug("Done loading model and preparing classification layer")
