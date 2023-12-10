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
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.logger.debug("Done loading model and preparing classification layer")
