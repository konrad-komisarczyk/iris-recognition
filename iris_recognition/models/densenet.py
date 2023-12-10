from __future__ import annotations

from torch import nn
from torchvision.models import densenet121

from iris_recognition.models.model import Model


class Densenet(Model):
    """
    Pretrained DenseNet model class
    """

    @property
    def name(self) -> str:
        return "DenseNet"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = densenet121(weights='DEFAULT')
        # in the original model classifier had more layers - is replacing it with 1 layer a good idea?
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        self.logger.debug("Done loading model and preparing classification layer")
