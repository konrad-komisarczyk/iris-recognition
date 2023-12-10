from __future__ import annotations

import torch
from torchvision.models import vgg16

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel


class VggPretrained(PretrainedModel):
    """
    Pretrained VGG16 model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)

        self.logger.debug("Loading model")
        self.model = vgg16(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "VGG16"

    def prepare_classification_layers(self, num_classes: int) -> None:
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, num_classes)
