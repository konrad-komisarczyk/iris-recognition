import random
import numpy as np
import matplotlib
import splitfolders
from PIL import Image
import os
import pandas as pd
import umap
import matplotlib.pyplot as plt
import shutil
from itertools import combinations
from tqdm import tqdm


def calculate_dispersion(group):
    return np.sqrt(np.var(group[0]) + np.var(group[1]))  # If created manually
    # return np.sqrt(np.var(group['0']) + np.var(group['1']))  # If read from csv file


def dist(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_mean_distance(group):
    points = list(zip(group[0], group[1]))  # If created manually
    # points = list(zip(group['0'], group['1']))  # If read from file
    distances = [dist(p1, p2) for p1, p2 in combinations(points, 2)]
    avg_distance = sum(distances) / len(distances)
    return avg_distance


def filter_from_datasets_and_copy_to_final_destination(datasets):
    good_classes = 0
    cmap = matplotlib.colormaps['Paired']
    mycmap = cmap(np.linspace(0, 1, 10))

    eye_id = 1
    for dataset in tqdm(datasets):
        root_dir = f'../{dataset}'

        image_data = []
        folders = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            if os.path.isdir(folder_path):
                if len(os.listdir(folder_path)) >= 5:
                    for file in os.listdir(folder_path):
                        if file.endswith('.png'):
                            file_path = os.path.join(folder_path, file)

                            img = Image.open(file_path).convert('L')
                            img_array = np.array(img)

                            flattened_img_array = img_array.flatten()

                            image_data.append(flattened_img_array)
                            folders.append(folder)

        # Create dataframe and save to file (takes a long time)
        df = pd.DataFrame(image_data)
        # df.to_csv("image_df.csv")

        # # Read dataframe from file
        # df = pd.read_csv("image_df.csv")
        # df.drop("Unnamed: 0", axis='columns', inplace=True)

        df['Class'] = folders
        df['Class'] = df["Class"].astype('int32')
        df.dropna(inplace=True)

        # UMAP
        reducer = umap.UMAP()
        reducer.fit(df.iloc[:, :-1])

        # Create all embeddings and save to file
        embedding_all = reducer.transform(df.iloc[:, :-1])
        emb_df_all = pd.DataFrame(embedding_all)
        emb_df_all["Class"] = df['Class'].tolist()
        # emb_df_all.to_csv("embedding_df_all.csv")

        # # Read all embeddings from file
        # emb_df_all = pd.read_csv("embedding_df_all.csv")
        # emb_df_all.drop("Unnamed: 0", axis='columns', inplace=True)

        dispersion_per_class = emb_df_all.groupby('Class').apply(calculate_dispersion)

        dispersion_df_all = pd.DataFrame({
            'class': dispersion_per_class.index,
            'dispersion': dispersion_per_class.values
        })

        print(dispersion_df_all.describe())

        mean_distance = emb_df_all.groupby('Class').apply(calculate_mean_distance)

        mean_distance_df_all = pd.DataFrame({
            'class': mean_distance.index,
            'mean_distance': mean_distance.values
        })

        print(mean_distance_df_all.describe())

        if dataset == "ubiris_filtered":
            mask_good_classes_dist = mean_distance_df_all[
                (mean_distance_df_all["mean_distance"] < mean_distance_df_all.quantile(0.1)['mean_distance'])][
                "class"].unique().tolist()
            mask_good_classes_dispersion = \
                dispersion_df_all[
                    (dispersion_df_all["dispersion"] < dispersion_df_all.quantile(0.1)['dispersion'])][
                    "class"].unique().tolist()
            mask_good = set(mask_good_classes_dist).intersection(set(mask_good_classes_dispersion))
        elif dataset == "miche_filtered":
            mask_good_classes_dist = mean_distance_df_all[
                (mean_distance_df_all["mean_distance"] < mean_distance_df_all.quantile(0.45)['mean_distance'])][
                "class"].unique().tolist()
            mask_good_classes_dispersion = \
                dispersion_df_all[
                    (dispersion_df_all["dispersion"] < dispersion_df_all.quantile(0.45)['dispersion'])][
                    "class"].unique().tolist()
            mask_good = set(mask_good_classes_dist).intersection(set(mask_good_classes_dispersion))
        else:  # dataset == "mmu_filtered"
            mask_good_classes_dist = mean_distance_df_all[
                (mean_distance_df_all["mean_distance"] < mean_distance_df_all.quantile(0.85)['mean_distance'])][
                "class"].unique().tolist()
            mask_good_classes_dispersion = \
                dispersion_df_all[
                    (dispersion_df_all["dispersion"] < dispersion_df_all.quantile(0.85)['dispersion'])][
                    "class"].unique().tolist()
            mask_good = set(mask_good_classes_dist).intersection(set(mask_good_classes_dispersion))

        print(f"Number of good classes: {len(mask_good)}")
        good_classes += len(mask_good)

        df_good = df.loc[df["Class"].isin(mask_good)]

        embedding_good = reducer.transform(df_good.iloc[:, :-1])

        emb_df_good = pd.DataFrame(embedding_good)
        emb_df_good["Class"] = df_good['Class'].tolist()

        colors = df_good["Class"]
        classes_unique = list(set(colors))

        chunks = [classes_unique[x:x + 10] for x in range(0, len(classes_unique), 10)]
        for chunk in chunks:
            cur_emb_df = emb_df_good.loc[emb_df_good["Class"].isin(chunk)]
            scatter_list = []
            label_list = []
            for num, color in enumerate(chunk, start=0):
                scatter = plt.scatter(cur_emb_df.loc[cur_emb_df["Class"].isin([color])][0],
                                      cur_emb_df.loc[cur_emb_df["Class"].isin([color])][1], color=mycmap[num], s=10)
                scatter_list.append(scatter)
                label_list.append(str(color))
            plt.gca().set_aspect('equal', 'datalim')
            plt.title(dataset)
            lgnd = plt.legend(scatter_list, label_list, bbox_to_anchor=(1, 0.9))
            for lgnhandle in lgnd.legend_handles:
                lgnhandle._sizes = [40]

            plt.show()

        cur_data_path = os.path.join(base_dir, dataset)
        for eye_dir in os.listdir(os.path.join(cur_data_path)):
            if int(eye_dir.split("/")[-1]) in mask_good:
                cur_eye_dest_dir = os.path.join(dest_data_path_before_split, str(eye_id))
                os.makedirs(cur_eye_dest_dir, exist_ok=True)
                for file in os.listdir(os.path.join(cur_data_path, eye_dir)):
                    shutil.copy(os.path.join(cur_data_path, eye_dir, file), cur_eye_dest_dir)

                eye_id += 1
    print(f"Total number of good classes {good_classes}")


base_dir = os.path.abspath(os.path.join('', '..'))
data_dir = os.path.join(base_dir, "data")
dest_data_path_before_split = os.path.join(data_dir, "umap")
dest_data_dir = os.path.join(data_dir, "datasets_preprocessed")

filter_from_datasets_and_copy_to_final_destination(["mmu_filtered", "ubiris_filtered", "miche_filtered"])

test_destination_dir = os.path.join(base_dir, f"umap_testing_sample")
os.makedirs(test_destination_dir, exist_ok=True)

input_dir = dest_data_path_before_split
classes_list = os.listdir(input_dir)
classes_sample = random.sample(classes_list, int(0.1 * len(classes_list)))

for sample_class in classes_sample:
    shutil.move(os.path.join(input_dir, sample_class), test_destination_dir)

output_dir = os.path.join(base_dir, "data", "datasets_preprocessed")
os.makedirs(output_dir, exist_ok=True)
splitfolders.ratio(input=input_dir, output=output_dir, ratio=(0.8, 0.2, 0.0), seed=3000)

os.rename(os.path.join(base_dir, "data", "datasets_preprocessed", "train"),
          os.path.join(base_dir, "data", "datasets_preprocessed", "umap_train"))
os.rename(os.path.join(base_dir, "data", "datasets_preprocessed", "val"),
          os.path.join(base_dir, "data", "datasets_preprocessed", "umap_val"))

shutil.rmtree(os.path.join(output_dir, "test"))
