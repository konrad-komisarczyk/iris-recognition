from iris_recognition import finetune

args = [
    "--model", "AlexNet",
    "--learning_rate", "0.00003",
    "--batch_size", "64",
    "--training_datasets",  "mmu_filtered_train",
    "--validation_datasets",  "mmu_filtered_val",
    "--num_epochs", "10",
    "--tag", "test",
]
finetune.main(args)

