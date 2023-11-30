from typing import Any

from iris_recognition.pretrained_models.pretrained_model import PretrainedModel
from iris_recognition.pretrained_models.pretrained_model import TrainingParams
from iris_recognition.trainset import Trainset


class DensenetPretrained(PretrainedModel):
    """
    Pretrained DenseNet model class
    """

    @property
    def name(self) -> str:
        return "DenseNet"

    def train(self, trainset: Trainset, params: TrainingParams) -> None:
        raise NotImplementedError()

    def save_as_finetuned(self, tag: str) -> None:
        raise NotImplementedError()

    def get_transform(self) -> Any:
        raise NotImplementedError()
