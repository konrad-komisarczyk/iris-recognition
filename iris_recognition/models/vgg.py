from __future__ import annotations

from torch import nn
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

        # Freeze all the layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        self.logger.debug("Done loading model and preparing classification layer")
