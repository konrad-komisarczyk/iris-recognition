import os
from get_iris import get_iris
from normalize import normalize
import subprocess
from tqdm import tqdm
import shutil
import time
from PIL import Image, ImageOps

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir1 = os.path.join(base_dir, "miche_iPhone5", "iPhone5")
data_dir2 = os.path.join(base_dir, "miche_SamsungGalaxyTab2", "SamsungGalaxyTab2")
data_dir3 = os.path.join(base_dir, "miche_SamsungGalaxyS4", "SamsungGalaxyS4")

destination_dir = os.path.join(base_dir, "miche_preprocessed")

exception_count = 0

# Select only the images taken with the rear camera (if available) and inside (outside images have huge reflections)

image_list = []
for img in os.listdir(data_dir1):
    if img[-3:] == "jpg":
        img_args = img.split("_")
        if "IN" in img_args and "R" in img_args:
            image_list.append(img)

for img in os.listdir(data_dir2):
    if img[-3:] == "jpg":
        img_args = img.split("_")
        if "IN" in img_args:
            image_list.append(img)

for img in os.listdir(data_dir3):
    if img[-3:] == "jpg":
        img_args = img.split("_")
        if "IN" in img_args and "R" in img_args:
            image_list.append(img)

start = time.time()
for img in tqdm(image_list):
    img_args = img.split("_")
    cur_dest_dir = os.path.join(destination_dir, str(img_args[0]))
    if not os.path.isdir(cur_dest_dir):
        os.mkdir(cur_dest_dir)

    if "IP5" in img:
        data_dir = data_dir1
    if "GT2" in img:
        data_dir = data_dir2
    else:
        data_dir = data_dir3

        # 3 lines below fix the problem of images taken with SamsungGalaxyS4
        # for some reason, they were being rotated by 90 degrees and segmentation didn't work
        im = Image.open(os.path.join(data_dir, img))
        im = ImageOps.exif_transpose(im)
        im.save(os.path.join(data_dir, img))

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
        # print(exception_count)

    file_name = img.split(".")[0]
    files_to_delete = [os.path.join(cur_dest_dir, f"{file_name}_inner_boundary.png"),
                       os.path.join(cur_dest_dir, f"{file_name}_outer_boundary.png"),
                       os.path.join(cur_dest_dir, f"{file_name}_iris.png"),
                       os.path.join(cur_dest_dir, f"{file_name}_seg.png"),
                       os.path.join(cur_dest_dir, f"{file_name}.jpg")]
    for file in files_to_delete:
        try:
            os.remove(file)
        except:
            pass

print(f"Was unable to preprocess {exception_count} images")
stop = time.time()

print(stop-start)