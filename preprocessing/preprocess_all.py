import os
from get_iris import get_iris
from normalize import normalize
import subprocess

base_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
data_dir = os.path.join(base_dir, "archive", "MMU-Iris-Database")

for folder in os.listdir(data_dir):
    print(f"Staring folder {folder}")
    if os.path.isdir(os.path.join(data_dir, folder)):
        cur_folder_path = os.path.join(data_dir, folder)
        for side in os.listdir(cur_folder_path):
            single_eye_path = os.path.join(cur_folder_path, side)
            for img in os.listdir(single_eye_path):
                if img[-3:] == "bmp":

                    subprocess.run(["python", "predict_one_img.py", "--gpu", "0", "--img_path", os.path.join(single_eye_path, img),"--save_dir", single_eye_path, "--model", "M1"])

                    file_name = img.split(".")[0]
                    get_iris(file_name, single_eye_path)
                    normalize(file_name, single_eye_path)

                    files_to_delete = [os.path.join(single_eye_path, f"{file_name}_inner_boundary.png"),
                                       os.path.join(single_eye_path, f"{file_name}_outer_boundary.png"),
                                       os.path.join(single_eye_path, f"{file_name}_iris.png"),
                                       os.path.join(single_eye_path, f"{file_name}_seg.png")]

                    for file in files_to_delete:
                        os.remove(file)