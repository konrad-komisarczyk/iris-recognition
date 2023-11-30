# `data`  directory

This is directory for storing data that is not uploaded to the github.

## `datasets_preprocessed`

Normalized iris images used for finetuning models and training.

Structure:

`{dataset_name}/{train|test}/{example_number}/{image}.png`

* `dataset_name`: `miche` / `mmu` / `ubiris`
* there are multiple images for an `example_number` - all from the same eye of the same person
* right and left eye of the same person are different examples
* all images have to be .png
* train/test split is 4:1

Datasets are stored on our Drive (`datasets_preprocessed.zip`).

## `finetuned_models`

Finetuning experiments results.

Structure:

`{model_name, example: resnet152}/{tag}/model_files`

tag is provided manually for a finetuning experiment
