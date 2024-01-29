from __future__ import annotations

import os.path
import pathlib


class PathOrganizer:
    """
    Class for organizing paths to project's resources
    """

    def __init__(self, prefix: str | None = None) -> None:
        self.prefix = prefix or PathOrganizer.get_root()

    @staticmethod
    def get_root() -> str:
        """
        :return: path to repo root
        """
        return str(pathlib.Path(__file__).parent.parent.parent.absolute())

    def get_segmentation_model_path(self) -> str:
        """
        :return: path where segmentation model should be copied
        """
        return os.path.join(self.prefix, "preprocessing", "trained_models")

    def get_dataset_preprocessed(self, dataset_name: str) -> str:
        """
        :param dataset_name: dataset name
        :return: path to the preprocessed dataset
        """
        return os.path.join(self.prefix, "data", "datasets_preprocessed", dataset_name)

    def get_finetuned_model_dir(self, model_name: str, tag: str) -> str:
        """
        :param model_name: model name
        :param tag: model's training tag
        :return: path to the finetuned model dir
        """
        return os.path.join(self.prefix, "data", "finetuned_models", model_name, tag)

    def get_finetuned_model_metrics_path(self, model_name: str, tag: str) -> str:
        """
        :param model_name: model name
        :param tag: model's training tag
        :return: path to the finetuned model metrics dump path
        """
        return os.path.join(self.get_finetuned_model_dir(model_name, tag), "metrics.txt")

    @staticmethod
    def get_finetuned_model_filename(epoch: int) -> str:
        """
        :param epoch: number of epoch
        :return: filename the finetuned model
        """
        return f"epoch{epoch}.pt"

    def get_finetuned_model_path(self, model_name: str, tag: str, epoch: int) -> str:
        """
        :param model_name: model name
        :param tag: model's training tag
        :param epoch: number of epoch
        :return: path to the finetuned model
        """
        return os.path.join(self.get_finetuned_model_dir(model_name, tag), self.get_finetuned_model_filename(epoch))

    def get_finetuning_log_path(self, model_name: str, tag: str) -> str:
        """
        :param model_name: model name
        :param tag: model's training tag
        :return: path to the finetuned model logs file
        """
        return os.path.join(self.get_finetuned_model_dir(model_name, tag), "model.log")

    def get_finetuning_params_dump_path(self, model_name: str, tag: str) -> str:
        """
        :param model_name: model name
        :param tag: model's training tag
        :return: path to the finetuned model logs file
        """
        return os.path.join(self.get_finetuned_model_dir(model_name, tag), "params.json")
