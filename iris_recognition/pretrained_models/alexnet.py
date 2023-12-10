from __future__ import annotations

from torch import nn
from torchvision.models import alexnet

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel


class AlexnetPretrained(PretrainedModel):
    """
    Pretrained AlexNet model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)
        self.model = alexnet(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "AlexNet"

    def prepare_classification_layers(self, num_classes: int) -> None:
        # in the original model classifier had more layers - is replacing it with 1 layer a good idea?
        self.model.classifier = nn.Linear(256 * 6 * 6, num_classes)
