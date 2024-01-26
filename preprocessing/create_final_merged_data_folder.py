import os
import shutil
import splitfolders
from tqdm import tqdm
import random


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
destination_dir = os.path.join(base_dir, "data", "all_filtered")
os.makedirs(destination_dir, exist_ok=True)

data_folders = ["miche_filtered", "ubiris_filtered", "mmu_filtered"]

eye_id = 1
for source in tqdm(data_folders):
    cur_data_path = os.path.join(base_dir, source)
    for eye_dir in os.listdir(os.path.join(cur_data_path)):
        if len(os.listdir(os.path.join(cur_data_path, eye_dir))) >= 5:
            cur_eye_dest_dir = os.path.join(destination_dir, str(eye_id))
            os.makedirs(cur_eye_dest_dir, exist_ok=True)
            for file in os.listdir(os.path.join(cur_data_path, eye_dir)):
                shutil.copy(os.path.join(cur_data_path, eye_dir, file), cur_eye_dest_dir)
            eye_id += 1


input_dir = os.path.join(base_dir, "data", "all_filtered")

test_destination_dir = os.path.join(base_dir, f"all_filtered_testing_sample")
os.makedirs(test_destination_dir, exist_ok=True)

classes_list = os.listdir(input_dir)
classes_sample = random.sample(classes_list, int(0.1 * len(classes_list)))

for sample_class in classes_sample:
    shutil.move(os.path.join(input_dir, sample_class), test_destination_dir)

output_dir = os.path.join(base_dir, "data", "datasets_preprocessed")
os.makedirs(output_dir, exist_ok=True)
splitfolders.ratio(input=input_dir, output=output_dir, ratio=(0.8, 0.2, 0.0), seed=3000)

os.rename(os.path.join(base_dir, "data", "datasets_preprocessed", "train"),
          os.path.join(base_dir, "data", "datasets_preprocessed", "all_filtered_train"))
os.rename(os.path.join(base_dir, "data", "datasets_preprocessed", "val"),
          os.path.join(base_dir, "data", "datasets_preprocessed", "all_filtered_val"))

shutil.rmtree(os.path.join(output_dir, "test"))

