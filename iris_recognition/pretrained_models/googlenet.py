from __future__ import annotations

from torch import nn
from torchvision.models import googlenet

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel


class GooglenetPretrained(PretrainedModel):
    """
    Pretrained GoogLeNet model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)
        self.model = googlenet(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "GoogLeNet"

    def prepare_classification_layers(self, num_classes: int) -> None:
        # in the original model classifier had more layers - is replacing it with 1 layer a good idea?
        self.model.fc = nn.Linear(1024, num_classes)
