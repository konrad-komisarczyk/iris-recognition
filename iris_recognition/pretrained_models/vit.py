from __future__ import annotations

from torch import nn
from torchvision.models import vit_b_16

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel


class VitPretrained(PretrainedModel):
    """
    Pretrained ViT model class
    """

    def __init__(self, prefix: str | None = None) -> None:
        super().__init__(prefix)
        self.model = vit_b_16(weights='DEFAULT')

    @property
    def name(self) -> str:
        return "ViT"

    def prepare_classification_layers(self, num_classes: int) -> None:
        self.model.heads.head = nn.Linear(self.model.hidden_dim, num_classes)
