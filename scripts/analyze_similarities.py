from __future__ import annotations

import itertools
import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import pandas as pd

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.matchers.cosine_similarity_matcher import CosineSimilarityMatcher
from iris_recognition.matchers.euclidean_distance_matcher import EuclideanDistanceMatcher
from iris_recognition.matchers.matcher import MATCHER_SIMILARITY_FUNCTION
from iris_recognition.models import get_model_by_name
from iris_recognition.tools.fs_tools import FsTools
from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer
from iris_recognition.trainset import Trainset

MODELS_TAGS_NODES = [("AlexNet", "mmu2", "features.12")]
DATASETS = ["umap_filtered_val"]
TRAINSET_LEN_LIMIT = 100
SIMILARITY_FUNC: MATCHER_SIMILARITY_FUNCTION = EuclideanDistanceMatcher.distance
SIMILARITY_NAME: str = "Euclidean Distance"

LOGGER = get_logger("Analyze similarities")


def similarities_distribution_info(similarities: list[float]) -> str:
    if not similarities:
        return "NO SIMILARITIES"
    return f"(min, median, max): ({min(similarities):.3f}, {median(similarities):.3f}, {max(similarities):.3f})"


for model_name, tag, node_name in MODELS_TAGS_NODES:
    model = get_model_by_name(model_name)
    model.load_finetuned(tag)
    LOGGER.info(f"Testing model: {model_name} from tag {tag}, node: {node_name}.")
    model.log_node_names()
    trainset = Trainset.load_dataset(DATASETS, None, TRAINSET_LEN_LIMIT)

    label_to_features: dict[str, list[ExtractedFeatures]] = defaultdict(list)
    features: ExtractedFeatures | None = None
    for i in range(len(trainset)):
        image, label = trainset[i]
        LOGGER.info(f"Extracting features from image {i} with label {label}.")
        features = model.extract_features(node_name, image)
        label_to_features[label].append(features)
    LOGGER.info("Done extracting features")
    LOGGER.info(f"Features shape {features.shape()}")

    in_label_similarities: dict[str, list[float]] = defaultdict(list)
    for label, label_features in label_to_features.items():
        LOGGER.info(f"Calculating in-label similarities for label {label}")
        for features1, features2 in itertools.combinations(label_features, 2):
            in_label_similarities[label].append(SIMILARITY_FUNC(features1, features2))
        LOGGER.info(f"Similarities dist. for label {label}: "
                    f"{similarities_distribution_info(in_label_similarities[label])}")
    all_in_label_similarities = list(itertools.chain.from_iterable(in_label_similarities.values()))

    between_label_similarities: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (label1, label1_features), (label2, label2_features) in itertools.combinations(label_to_features.items(), 2):
        LOGGER.info(f"Calculating between-label similarities for labels {label1} - {label2}")
        for features1, features2 in itertools.product(label1_features, label2_features):
            between_label_similarities[(label1, label2)].append(SIMILARITY_FUNC(features1, features2))
        LOGGER.info(f"Similarities dist. for pair {label1} - {label2}: "
                    f"{similarities_distribution_info(between_label_similarities[(label1, label2)])}")
    all_between_label_similarities = list(itertools.chain.from_iterable(between_label_similarities.values()))

    # plotting densities
    df_dict = {
        'is_inlabel': ["in_label"] * len(all_in_label_similarities) + ["between"] * len(all_between_label_similarities),
        'similarity': all_in_label_similarities + all_between_label_similarities
    }
    df = pd.DataFrame.from_dict(df_dict)
    df.groupby(df.is_inlabel).similarity.plot.kde()
    plt.legend()
    datasets_name_joined = ','.join(DATASETS)
    plt.title(f"{model_name} {tag} {node_name} on sets: {datasets_name_joined}")
    plt.suptitle(SIMILARITY_NAME)
    histogram_path = os.path.join(PathOrganizer.get_root(), "similarities_plots", SIMILARITY_NAME,
                                  f"{model_name}-{tag}-{node_name}-{datasets_name_joined}.png")
    FsTools.ensure_dir(histogram_path)
    plt.tight_layout()
    plt.savefig(histogram_path)
    LOGGER.info(f"Done. Plot saved to {histogram_path}.")
