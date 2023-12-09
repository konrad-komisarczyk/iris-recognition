import os
import random
import shutil


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, "all_preprocessed_filtered")
destination_dir = os.path.join(base_dir, "final_testing_sample")
os.makedirs(destination_dir, exist_ok=True)

classes_list = os.listdir(data_dir)
classes_sample = random.sample(classes_list, int(0.1*len(classes_list)))

for sample_class in classes_sample:
    shutil.move(os.path.join(data_dir, sample_class), destination_dir)
