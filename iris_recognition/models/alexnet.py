from __future__ import annotations

from torch import nn
from torchvision.models import alexnet

from iris_recognition.models.model import Model


class Alexnet(Model):
    """
    Pretrained AlexNet model class
    """

    @property
    def name(self) -> str:
        return "AlexNet"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = alexnet(weights='DEFAULT')
        num_ftrs = self.model.classifier[1].in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # Final layer for num_classes classes
        )
        self.logger.debug("Done loading model and preparing classification layer")
