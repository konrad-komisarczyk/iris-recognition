import os
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
data_dir = os.path.join(base_dir, "archive", "MMU-Iris-Database")
destination_dir = os.path.join(base_dir, "mmu_preprocessed")
os.makedirs(destination_dir, exist_ok=True)

eye_id = 1
for folder in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, folder)):
        cur_folder_path = os.path.join(data_dir, folder)
        for side in os.listdir(cur_folder_path):
            single_eye_path = os.path.join(cur_folder_path, side)
            os.makedirs(os.path.join(destination_dir, str(eye_id)), exist_ok=True)
            for img in os.listdir(single_eye_path):
                if img[-14:] == 'normalized.png':
                    shutil.copy(os.path.join(single_eye_path, img), os.path.join(destination_dir, str(eye_id)))
            eye_id += 1
