from __future__ import annotations

from torchvision.models import resnet152
import torch

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel


class ResnetPretrained(PretrainedModel):
    """
    Pretrained resnet152 model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)

        self.logger.debug("Loading model")
        self.model = resnet152(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "resnet152"

    def prepare_classification_layers(self, num_classes: int) -> None:
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
