import random
import os
import shutil
import splitfolders
from tqdm import tqdm

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for dataset in ["all_filtered", "ubiris_filtered"]:
    data_dir = os.path.join(base_dir, "data", dataset)
    destination_dir = os.path.join(base_dir, "data", f"{dataset}_undersampled")
    os.makedirs(destination_dir, exist_ok=True)

    eye_id = 1
    for cls in tqdm(os.listdir(data_dir)):
        class_list = os.listdir(os.path.join(data_dir, cls))
        class_size = len(class_list)
        if class_size >= 5:
            classes_sample = random.sample(class_list, class_size - 5) if class_size > 5 else []

            samples_to_copy = list(set(class_list).difference(set(classes_sample)))
            cur_eye_dest_dir_train = os.path.join(destination_dir, str(eye_id))
            for sample in samples_to_copy:
                os.makedirs(cur_eye_dest_dir_train, exist_ok=True)
                shutil.copy(os.path.join(data_dir, cls, sample), cur_eye_dest_dir_train)
            eye_id += 1

    input_dir = destination_dir

    test_destination_dir = os.path.join(base_dir, f"{dataset}_undersampled_testing_sample")
    os.makedirs(test_destination_dir, exist_ok=True)

    classes_list = os.listdir(input_dir)
    classes_sample = random.sample(classes_list, int(0.1 * len(classes_list)))

    for sample_class in classes_sample:
        shutil.move(os.path.join(input_dir, sample_class), test_destination_dir)

    output_dir = os.path.join(base_dir, "data", "datasets_preprocessed")
    os.makedirs(output_dir, exist_ok=True)
    splitfolders.ratio(input=input_dir, output=output_dir, ratio=(0.8, 0.2, 0.0), seed=3000)

    os.rename(os.path.join(base_dir, "data", "datasets_preprocessed", "train"),
              os.path.join(base_dir, "data", "datasets_preprocessed", f"{dataset}_undersampled_train"))
    os.rename(os.path.join(base_dir, "data", "datasets_preprocessed", "val"),
              os.path.join(base_dir, "data", "datasets_preprocessed", f"{dataset}_undersampled_val"))

    shutil.rmtree(os.path.join(output_dir, "test"))
