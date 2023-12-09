import os
import shutil
from tqdm import tqdm

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
destination_dir = os.path.join(base_dir, "all_preprocessed_filtered")
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
