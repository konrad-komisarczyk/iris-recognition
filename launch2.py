from iris_recognition import finetune

args = [
    "--model", "resnet152",
    "--learning_rate", "0.00003",
    "--batch_size", "64",
    "--training_datasets",  "cifar_train",
    "--validation_datasets",  "cifar_val",
    "--num_epochs", "1",
    "--tag", "cifar1",
]
finetune.main(args)
