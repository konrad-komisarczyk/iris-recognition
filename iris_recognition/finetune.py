from __future__ import annotations

import argparse
import logging
import sys

from iris_recognition.models import model_name_to_class, get_model_by_name
from iris_recognition.models.model import TrainingParams
from iris_recognition.tools.logger import set_loggers_stderr_verbosity
from iris_recognition.trainset import AVAILABLE_DATASETS, Trainset


def get_parser() -> argparse.ArgumentParser:
    """
    :return: argv parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(model_name_to_class.keys()),
                        help=f"Name of the model to train, choose one of {list(model_name_to_class.keys())}")
    parser.add_argument("--training_datasets", type=str, required=True, nargs="+",
                        choices=AVAILABLE_DATASETS,
                        help=f"Names of the sets to include to training set, choose from {AVAILABLE_DATASETS}")
    parser.add_argument("--validation_datasets", type=str, required=False, nargs="+",
                        choices=AVAILABLE_DATASETS,
                        help=f"Optional, names of the sets to include to validation set, "
                             f"choose from {AVAILABLE_DATASETS}")
    parser.add_argument("--trainset_len_limit", type=int, required=False,
                        help="Optional limit to trainset len")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs, default is 1")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size, default is 1")
    parser.add_argument("--learning_rate", type=int, default=0.001,
                        help="Learning rate training param")
    parser.add_argument("--weight_decay", type=int, default=0.0001,
                        help="Weight decay training param")
    parser.add_argument("--tag", type=str, required=True,
                        help="Training tag, example: date")
    return parser


def finetune(parsed_args: argparse.Namespace) -> None:
    """
    :param parsed_args: parsed args
    """
    training_params = TrainingParams(num_epochs=parsed_args.num_epochs, learning_rate=parsed_args.learning_rate,
                                     weight_decay=parsed_args.weight_decay, batch_size=parsed_args.batch_size)
    model = get_model_by_name(parsed_args.model)
    transform = model.get_transform()
    trainset = Trainset.load_dataset(parsed_args.training_datasets, transform, parsed_args.trainset_len_limit)
    valset = Trainset.load_dataset(parsed_args.validation_datasets, transform, parsed_args.trainset_len_limit) \
        if parsed_args.validation_datasets else None
    model.prepare_pretrained(trainset.num_classes())
    model.train(trainset, valset, training_params, tag_to_save=parsed_args.tag)

    model.log_node_names()


def main(args: list[str] | None) -> None:
    parsed_args = get_parser().parse_args(args)
    finetune(parsed_args)


if __name__ == '__main__':
    set_loggers_stderr_verbosity(logging.DEBUG)
    main(sys.argv[1:])
