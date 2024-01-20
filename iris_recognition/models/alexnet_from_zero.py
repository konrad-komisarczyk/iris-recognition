from __future__ import annotations

from torch import nn
from torchvision.models import alexnet

from iris_recognition.models.model import Model


class AlexnetFromZero(Model):
    """
    Pretrained AlexNet model class
    """

    @property
    def name(self) -> str:
        return "AlexNetFromZero"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = alexnet(weights=None)
        self.logger.debug("Done loading model and preparing classification layer")
