from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer


class Trainset(Dataset):
    """
    Class for loading data for finetuning pretrained_models
    """
    SEED = 213

    def __init__(self, transform: transforms.Compose | None) -> None:
        """
        :param transform: transform function to be applied to images, or None if no transform should be applied
        """
        np.random.seed(Trainset.SEED)
        self.transform = transform
        self.image_paths: list[str] = []
        self.labels: list[int] = []  # example: [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 4, ...]
        self.label_names: list[str] = []  # self.label_names[i] = name of the label i

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

    def get_dataloader(self, batch_size: int = 1) -> DataLoader:
        """
        :return: DataLoader
        :param batch_size: batch size, default 1
        """
        return DataLoader(self, batch_size=batch_size)

    @staticmethod
    def load_dataset(dataset_names: list[str], transform: Any, limit_examples: int | None = None,
                     example_names_to_keep: set[str] | None = None) -> Trainset:
        """
        :param dataset_names: list of names of datasets to load from
        :param transform: transform function that will be applied to images
        :param limit_examples: optional total examples limit
        :param example_names_to_keep: if not None, then load only examples with given names
        :return: Trainset with examples from all given datasets
        """
        path_organizer = PathOrganizer()
        res = Trainset(transform)
        res.logger.info(f"Loading Trainset examples...")
        for dataset_name in dataset_names:
            dataset_train_dir = path_organizer.get_dataset_preprocessed(dataset_name)
            res._load_examples_from_dir(dataset_train_dir, limit_examples, example_names_to_keep)
        res.logger.info(f"Finished loading Trainset examples. Trainset size: {len(res)}.\n"
                        f"Label_names: {res.label_names}.")
        return res

    def _load_examples_from_dir(self, dataset_train_dir: str, limit_examples: int | None = None,
                                example_names_to_keep: set[str] | None = None) -> None:
        self.logger.info(f"Loading examples for dataset from {dataset_train_dir}")
        subfolder_names = os.listdir(dataset_train_dir)
        if example_names_to_keep is not None:
            subfolder_names = example_names_to_keep.intersection(subfolder_names)
            self.logger.info(f"Limiting examples to {subfolder_names}")
        subfolders_paths = [os.path.join(dataset_train_dir, dirname) for dirname in subfolder_names]
        examples_paths = [dirpath for dirpath in subfolders_paths if os.path.isdir(dirpath)]
        examples_paths.sort()
        for example_path in examples_paths:
            self._load_persons_examples(example_path, limit_examples)
        self.logger.info(f"Finished loading examples for dataset from {dataset_train_dir}. "
                         f"Dataset current size {len(self)}")

    def _load_persons_examples(self, persons_dir: str, limit_examples: int | None = None) -> None:
        self.logger.debug(f"Loading images from dir {persons_dir}")
        new_label = self.num_classes()
        new_label_name = os.path.basename(persons_dir)
        images = [filename for filename in os.listdir(persons_dir) if filename.endswith((".png", ".jpg"))]
        self.logger.debug(f"Images found in dir: {images}")
        imagepaths = [os.path.join(persons_dir, filename) for filename in images]
        if (limit_examples is None or len(self) < limit_examples) and imagepaths:
            self.label_names.append(new_label_name)
        for imagepath in imagepaths:
            if limit_examples is None or len(self) < limit_examples:
                self.logger.debug(f"Adding to trainset image with label {new_label} from path {imagepath}")
                self.labels.append(new_label)
                self.image_paths.append(imagepath)
