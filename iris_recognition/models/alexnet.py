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
        # in the original model classifier had more layers - is replacing it with 1 layer a good idea?
        self.model.classifier = nn.Linear(256 * 6 * 6, num_classes)
        self.logger.debug("Done loading model and preparing classification layer")
