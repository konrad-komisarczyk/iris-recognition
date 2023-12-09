import os
from get_iris import get_iris
from normalize import normalize
import subprocess
from tqdm import tqdm
import shutil
import time


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, "ubiris2_1", "CLASSES_400_300_Part1")  # performed separately for both parts
destination_dir = os.path.join(base_dir, "ubiris2_1_preprocessed")  # performed separately for both parts
os.makedirs(destination_dir, exist_ok=True)
exception_count = 0

start = time.time()
for img in tqdm(sorted(os.listdir(data_dir))):
    if img[-4:] == "tiff":
        eye_id = int(img.split(".")[0].split("_")[0][1:])

        cur_dest_dir = os.path.join(destination_dir, str(eye_id))
        os.makedirs(cur_dest_dir, exist_ok=True)

        subprocess.run(
            ["python", "predict_one_img.py", "--gpu", "0", "--img_path", os.path.join(data_dir, img),
             "--save_dir", cur_dest_dir, "--model", "M1"])

        shutil.copy(os.path.join(data_dir, img), cur_dest_dir)

        try:
            get_iris(img, cur_dest_dir)
            # for resnet152 and densenet201
            normalize(img, cur_dest_dir, 56, 112)
        except Exception as e:
            exception_count += 1

        file_name = img.split(".")[0]
        files_to_delete = [os.path.join(cur_dest_dir, f"{file_name}_inner_boundary.png"),
                           os.path.join(cur_dest_dir, f"{file_name}_outer_boundary.png"),
                           os.path.join(cur_dest_dir, f"{file_name}_iris.png"),
                           os.path.join(cur_dest_dir, f"{file_name}_seg.png"),
                           os.path.join(cur_dest_dir, f"{file_name}.tiff")]

        for file in files_to_delete:
            try:
                os.remove(file)
            except:
                pass

print(f"Was unable to preprocess {exception_count} images")
stop = time.time()

print(stop-start)