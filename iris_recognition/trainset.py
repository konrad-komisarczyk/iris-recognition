from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer


AVAILABLE_DATASETS = ["miche", "mmu", "ubiris"]


class Trainset(Dataset):
    """
    Class for loading data for finetuning pretrained_models
    """
    SEED = 213

    def __init__(self, transform: Any, valid_size: float = 0.3, batch_size: int = 1) -> None:
        """
        :param transform: transform function to be applied to images
        :param valid_size: fraction of validation set, default 0.3
        :param batch_size: batch size, default 1
        """
        np.random.seed(Trainset.SEED)
        self.transform = transform
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.image_paths: list[str] = []
        self.labels: list[int] = []

        self.logger = get_logger("Trainset")

    def __len__(self):
        return len(self.image_paths)

    def num_classes(self) -> int:
        """
        :return: number of classes
        """
        return self.labels[-1] + 1 if self.labels else 0

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def train_len(self) -> int:
        return len(self) - self.valid_len()

    def valid_len(self) -> int:
        return int(np.floor(self.valid_size * len(self)))

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        :return: tuple[Train DataLoader, Validation Dataloader]
        """
        indices = list(range(len(self)))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[self.valid_len():], indices[:self.valid_len()]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(self, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(self, batch_size=self.batch_size, sampler=valid_sampler)
        return train_loader, valid_loader

    @staticmethod
    def load_dataset(dataset_names: list[str], transform: Any, limit_examples: int | None = None) -> Trainset:
        """
        :param dataset_names: list of names of datasets to load from
        :param transform: transform function that will be applied to images
        :param limit_examples: optional total examples limit
        :return: Trainset with examples from all given datasets
        """
        path_organizer = PathOrganizer()
        res = Trainset(transform)
        res.logger.info(f"Loading Trainset examples...")
        for dataset_name in dataset_names:
            dataset_train_dir = os.path.join(path_organizer.get_dataset_preprocessed(dataset_name), "train")
            res._load_examples_from_dir(dataset_train_dir, limit_examples)
        res.logger.info(f"Finished loading Trainset examples. Trainset size: {len(res)}")
        return res

    def _load_examples_from_dir(self, dataset_train_dir: str, limit_examples: int | None = None) -> None:
        self.logger.info(f"Loading examples for dataset from {dataset_train_dir}")
        subfolders_paths = [os.path.join(dataset_train_dir, dirname) for dirname in os.listdir(dataset_train_dir)]
        examples_paths = [dirpath for dirpath in subfolders_paths if os.path.isdir(dirpath)]
        for example_path in examples_paths:
            self._load_persons_examples(example_path, limit_examples)
        self.logger.info(f"Finished loading examples for dataset from {dataset_train_dir}. "
                         f"Dataset current size {len(self)}")

    def _load_persons_examples(self, persons_dir: str, limit_examples: int | None = None) -> None:
        self.logger.debug(f"Loading images from dir {persons_dir}")
        new_label = self.num_classes()
        images = [filename for filename in os.listdir(persons_dir) if filename.endswith(".png")]
        self.logger.debug(f"Images found in dir: {images}")
        imagepaths = [os.path.join(persons_dir, filename) for filename in images]
        for imagepath in imagepaths:
            if limit_examples is None or len(self) < limit_examples:
                self.logger.debug(f"Adding to trainset image with label {new_label} from path {imagepath}")
                self.labels.append(new_label)
                self.image_paths.append(imagepath)
