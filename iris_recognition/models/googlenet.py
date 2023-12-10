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
        self.model.fc = nn.Linear(1024, num_classes)
        self.logger.debug("Done loading model and preparing classification layer")
