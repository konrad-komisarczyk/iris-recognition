import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def filter_bad_segmentation_samples(data_dir, destination_dir, threshold):
    for eye in tqdm(os.listdir(data_dir)):
        eye_path = os.path.join(data_dir, eye)
        cur_dest_dir = os.path.join(destination_dir, eye)
        if not os.path.isdir(cur_dest_dir):
            os.mkdir(cur_dest_dir)
        for file in os.listdir(eye_path):
            img = Image.open(os.path.join(eye_path, file))
            arr = np.array(img)[:, :, 0]
            zero_num = np.count_nonzero(arr == 0)
            if ((arr.size - zero_num) / arr.size) * 100 > threshold:
                img.save(os.path.join(cur_dest_dir, file))


data_dir = os.path.join(base_dir, "ubiris_all_preprocessed")
destination_dir = os.path.join(base_dir, "ubiris_filtered")
os.makedirs(destination_dir, exist_ok=True)

filter_bad_segmentation_samples(data_dir, destination_dir, 60)

plt.hist([len(os.listdir(os.path.join(destination_dir, x))) for x in os.listdir(destination_dir)], bins=30)
good = sum([len(os.listdir(os.path.join(destination_dir, x))) >= 5 for x in os.listdir(destination_dir)])
plt.title(f"Ubiris filtered, >=5 num: {good}")
plt.show()

data_dir = os.path.join(base_dir, "miche_preprocessed")
destination_dir = os.path.join(base_dir, "miche_filtered")
os.makedirs(destination_dir, exist_ok=True)

filter_bad_segmentation_samples(data_dir, destination_dir, 60)

plt.hist([len(os.listdir(os.path.join(destination_dir, x))) for x in os.listdir(destination_dir)], bins=12)
good = sum([len(os.listdir(os.path.join(destination_dir, x))) >= 5 for x in os.listdir(destination_dir)])
plt.title(f"Miche filtered, >=5 num: {good}")
plt.show()

data_dir = os.path.join(base_dir, "mmu_preprocessed")
destination_dir = os.path.join(base_dir, "mmu_filtered")
os.makedirs(destination_dir, exist_ok=True)

filter_bad_segmentation_samples(data_dir, destination_dir, 60)

plt.hist([len(os.listdir(os.path.join(destination_dir, x))) for x in os.listdir(destination_dir)], bins=12)
good = sum([len(os.listdir(os.path.join(destination_dir, x))) >= 5 for x in os.listdir(destination_dir)])
plt.title(f"MMU filtered, >=5 num: {good}")
plt.show()
