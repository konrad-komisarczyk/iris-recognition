# `data`  directory

This is directory for storing data that is not uploaded to the github.

## `datasets_preprocessed`

Normalized iris images used for finetuning models and training.

Structure:

`{dataset_name}/{class_name}/{image}.png`

Datasets are stored on our Drive (`datasets_preprocessed.zip`).

## `finetuned_models`

Finetuning experiments results.

Structure:

`{model_name, example: resnet152}/{tag}/model_files`

tag is provided manually for a finetuning experiment

