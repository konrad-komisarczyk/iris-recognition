from __future__ import annotations

import torch
from torchvision.models import vgg16

from iris_recognition.models.model import Model


class Vgg(Model):
    """
    Pretrained VGG16 model class
    """

    @property
    def name(self) -> str:
        return "VGG16"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = vgg16(weights='DEFAULT')
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, num_classes)
        self.logger.debug("Done loading model and preparing classification layer")
