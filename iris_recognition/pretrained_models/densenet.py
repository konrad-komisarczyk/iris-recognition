from __future__ import annotations

from torch import nn
from torchvision.models import densenet121

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel


class DensenetPretrained(PretrainedModel):
    """
    Pretrained DenseNet model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)
        self.model = densenet121(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "DenseNet"

    def prepare_classification_layers(self, num_classes: int) -> None:
        # in the original model classifier had more layers - is replacing it with 1 layer a good idea?
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
