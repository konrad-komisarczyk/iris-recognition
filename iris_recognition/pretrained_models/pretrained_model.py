from __future__ import annotations

import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer
from iris_recognition.trainset import Trainset


@dataclass
class TrainingParams:
    """
    Class containing all settable training parameters
    """
    num_epochs: int
    learning_rate: float
    weight_decay: float


class PretrainedModel(abc.ABC):
    """
    Class representing pretrained model. Used to finetune model
    """

    def __init__(self, prefix: str | None = None) -> None:
        self.path_organizer = PathOrganizer(prefix)
        self.logger = get_logger("Pretrained Model " + self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: name of the model
        """

    @abstractmethod
    def train(self, trainset: Trainset, params: TrainingParams) -> None:
        """
        Train model
        """

    @abstractmethod
    def save_as_finetuned(self, tag: str) -> None:
        """
        Saves model to use in the testing
        """

    @abstractmethod
    def get_transform(self) -> Any:
        """
        :return: transform function that has to be applied to model's inputs
        """
