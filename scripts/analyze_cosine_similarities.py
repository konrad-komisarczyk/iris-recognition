import itertools
import os
import pathlib
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import pandas as pd

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.matchers.cosine_similarity_matcher import CosineSimilarityMatcher
from iris_recognition.models import get_model_by_name
from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer
from iris_recognition.trainset import Trainset

MODELS_TAGS_NODES = [("AlexNet", "test0", "features.12")]
DATASETS = ["miche", "mmu", "ubiris"]
TRAINSET_LEN_LIMIT = 100
HISTOGRAM_PATH = os.path.join(PathOrganizer.get_root(), "cosine_similarities.png")

SIMILARITY_FUNC = CosineSimilarityMatcher.similarity

LOGGER = get_logger("Analyze similarities")


def similarities_distribution_info(similarities: list[float]) -> str:
    if not similarities:
        return "NO SIMILARITIES"
    return f"(min, median, max): ({min(similarities):.3f}, {median(similarities):.3f}, {max(similarities):.3f})"


for model_name, tag, node_name in MODELS_TAGS_NODES:
    model = get_model_by_name(model_name)
    model.load_finetuned(tag)
    trainset = Trainset.load_dataset(DATASETS, None, TRAINSET_LEN_LIMIT)

    label_to_features: dict[str, list[ExtractedFeatures]] = defaultdict(list)
    for i in range(len(trainset)):
        image, label = trainset[i]
        LOGGER.info(f"Extracting features from image {i} with label {label}.")
        features = model.extract_features(node_name, image)
        label_to_features[label].append(features)
    LOGGER.info("Done extracting features")

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
        for features1 in label1_features:
            for features2 in label2_features:
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
    os.makedirs(pathlib.Path(HISTOGRAM_PATH).parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH)
