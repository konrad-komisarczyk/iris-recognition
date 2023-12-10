from __future__ import annotations

from torch import nn
from torchvision.models import vit_b_16

from iris_recognition.models.model import Model


class Vit(Model):
    """
    Pretrained ViT model class
    """

    @property
    def name(self) -> str:
        return "ViT"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = vit_b_16(weights='DEFAULT')
        self.model.heads.head = nn.Linear(self.model.hidden_dim, num_classes)
        self.logger.debug("Done loading model and preparing classification layer")
