from iris_recognition import finetune

args = [
    "--model", "AlexNet",
    "--learning_rate", "0.00003",
    "--batch_size", "64",
    "--training_datasets",  "umap_filtered_train",
    "--validation_datasets",  "umap_filtered_val",
    "--num_epochs", "10",
    "--tag", "testing_colab",
]
finetune.main(args)

