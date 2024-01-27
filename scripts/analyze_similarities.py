from __future__ import annotations

import itertools
import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.matchers.cosine_similarity_matcher import CosineSimilarityMatcher
from iris_recognition.matchers.euclidean_distance_matcher import EuclideanDistanceMatcher
from iris_recognition.matchers.matcher import MATCHER_SIMILARITY_FUNCTION
from iris_recognition.models import get_model_by_name, Model
from iris_recognition.tools.fs_tools import FsTools
from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer
from iris_recognition.irisdataset import IrisDataset

MODELS_TAGS_NODES = [("AlexNetFromZero", "mmu_all_best", "features.11")]
DATASETS = ["mmu_all_train", "mmu_all_val"]
#DATASETS = ["mmu_all_testing_sample"]
TRAINSET_LEN_LIMIT = None  # 100  # None

SIMILARITY_FUNC: MATCHER_SIMILARITY_FUNCTION = CosineSimilarityMatcher.similarity
SIMILARITY_NAME: str = "Podobieństwo cosinusowe"
# SIMILARITY_FUNC: MATCHER_SIMILARITY_FUNCTION = EuclideanDistanceMatcher.distance
# SIMILARITY_NAME: str = "Przeciwieństwo odległości euklidesowej"

#PLOT_TITLE: str = "treningowy + walidacyjny"
PLOT_TITLE: str = "features.11"
FIXED_BEST_THRESHOLD = None

# SIMILARITY_FUNC: MATCHER_SIMILARITY_FUNCTION = CosineSimilarityMatcher.similarity
# SIMILARITY_NAME: str = "Cosine Similarity"

LOGGER = get_logger("Analyze similarities")


def balanced_accuracy_for_threshold(threshold: float, all_in: list[float], all_between: list[float]) -> float:
    tp = sum(1 for sim in all_in if sim >= threshold)
    tn = sum(1 for sim in all_between if sim < threshold)
    sensitivity = tp / len(all_in)
    specificity = tn / len(all_between)
    return (sensitivity + specificity) / 2


def find_best_threshold(all_in: list[float], all_between: list[float]) -> tuple[float, float]:
    all_values = list(set(all_in) | set(all_between))
    best_threshold = 0
    best_value = -np.Inf
    for potential_threshold in all_values:
        #potential_threshold = (all_values[idx] + all_values[idx + 1]) / 2
        if (new_value := balanced_accuracy_for_threshold(potential_threshold, all_in, all_between)) >= best_value:
            best_value = new_value
            best_threshold = potential_threshold
    return best_threshold, best_value


def similarities_distribution_info(similarities: list[float]) -> str:
    if not similarities:
        return "NO SIMILARITIES"
    return f"(min, median, max): ({min(similarities):.3f}, {median(similarities):.3f}, {max(similarities):.3f})"


for model_name, tag, node_name in MODELS_TAGS_NODES:
    model = get_model_by_name(model_name)
    model.load_finetuned(tag)
    LOGGER.info(f"Testing model: {model_name} from tag {tag}, node: {node_name}.")
    model.log_node_names()
    trainset = IrisDataset.load_dataset(DATASETS, None, TRAINSET_LEN_LIMIT)

    LOGGER.info("Extracting features...")
    label_to_features: dict[str, list[ExtractedFeatures]] = defaultdict(list)
    features: ExtractedFeatures | None = None
    for i in range(len(trainset)):
        image, label = trainset[i]
        LOGGER.debug(f"Extracting features from image {i} with label {label}.")
        features = model.extract_features(node_name, image)
        label_to_features[label].append(features)
    LOGGER.info(f"Features shape {features.shape()}")

    LOGGER.info("Calculating in-label similarities...")
    in_label_similarities: dict[str, list[float]] = defaultdict(list)
    for label, label_features in label_to_features.items():
        LOGGER.debug(f"Calculating in-label similarities for label {label}")
        for features1, features2 in itertools.combinations(label_features, 2):
            in_label_similarities[label].append(SIMILARITY_FUNC(features1, features2))
        LOGGER.debug(f"Similarities dist. for label {label}: "
                     f"{similarities_distribution_info(in_label_similarities[label])}")
    all_in_label_similarities = list(itertools.chain.from_iterable(in_label_similarities.values()))

    LOGGER.info("Calculating between-label similarities...")
    between_label_similarities: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (label1, label1_features), (label2, label2_features) in itertools.combinations(label_to_features.items(), 2):
        LOGGER.debug(f"Calculating between-label similarities for labels {label1} - {label2}")
        for features1, features2 in itertools.product(label1_features, label2_features):
            between_label_similarities[(label1, label2)].append(SIMILARITY_FUNC(features1, features2))
        LOGGER.debug(f"Similarities dist. for pair {label1} - {label2}: "
                     f"{similarities_distribution_info(between_label_similarities[(label1, label2)])}")
    all_between_label_similarities = list(itertools.chain.from_iterable(between_label_similarities.values()))

    if FIXED_BEST_THRESHOLD is None:
        LOGGER.info("Searching for best threshold...")
        best_threshold, best_ba = find_best_threshold(all_in_label_similarities, all_between_label_similarities)
        LOGGER.info(f"Suggested best threshold: {best_threshold:.3f} gives balanced accuracy {best_ba:.3f}")
    else:
        best_threshold = FIXED_BEST_THRESHOLD

    # plotting densities
    df_dict = {
        'is_inlabel': ["różne klasy"] * len(all_between_label_similarities) +
                      ["ta sama klasa"] * len(all_in_label_similarities),
        'similarity': all_between_label_similarities +
                      all_in_label_similarities
    }
    df = pd.DataFrame.from_dict(df_dict)
    df.groupby(df.is_inlabel).similarity.plot.kde()

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 14
    datasets_name_joined = ','.join(DATASETS)
    plt.title(PLOT_TITLE)
    #plt.suptitle(SIMILARITY_NAME)
    plt.axvline(x=best_threshold, color="r", label="wyznaczony próg", linestyle="dashed")
    plt.ylabel("gęstość przybliżonego rozkładu")
    plt.xlabel("wartość metryki podobieństwa")
    plt.legend()
    histogram_path = os.path.join(PathOrganizer.get_root(), "similarities_plots", SIMILARITY_NAME,
                                  f"{model_name}-{tag}-{node_name}-{datasets_name_joined}.png")
    FsTools.ensure_dir(histogram_path)
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.show()
    LOGGER.info(f"Done. Plot saved to {histogram_path}.")
