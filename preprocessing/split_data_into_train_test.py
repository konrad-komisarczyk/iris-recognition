import os
import splitfolders
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset = "ubiris"  # "mmu"/"miche"/"ubiris"/
input_dir = os.path.join(base_dir, "data", dataset)  # "all_preprocessed_filtered" instead of dataset
output_dir = os.path.join(base_dir, "data", "datasets_preprocessed", dataset)  # "all_filtered" instead of dataset
os.makedirs(output_dir, exist_ok=True)
splitfolders.ratio(input=input_dir, output=output_dir, ratio=(0.8, 0.0, 0.2), seed=3000)

shutil.rmtree(os.path.join(output_dir, "val"))
