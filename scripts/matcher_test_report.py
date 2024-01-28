from __future__ import annotations

import itertools
import uuid
from collections import defaultdict

from tqdm import tqdm

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.matchers.cosine_similarity_matcher import CosineSimilarityMatcher
from iris_recognition.matchers.euclidean_distance_matcher import EuclideanDistanceMatcher
from iris_recognition.matchers.matcher import Matcher
from iris_recognition.models import get_model_by_name, Model
from iris_recognition.tools.fs_tools import FsTools
from iris_recognition.tools.logger import get_logger
from iris_recognition.irisdataset import IrisDataset
from iris_recognition.tools.path_organizer import PathOrganizer

HAVE_ENOUGH_RAM = False

LIMIT_EXAMPLES = None  # TODO: set to None later

MATCHERS: list[Matcher] = [CosineSimilarityMatcher(threshold=0.978)]
TESTSET_NAMES = ["mmu_all_testing_sample"]
MODELS_TAGS_NODES = [("GoogLeNet", "mmu_all_best", "avgpool")]

LOGGER = get_logger("Matcher test report")

testset = IrisDataset.load_dataset(TESTSET_NAMES, transform=None, limit_examples=LIMIT_EXAMPLES)


def load_features(path: str) -> ExtractedFeatures:
    with open(path, mode="rb") as f:
        return ExtractedFeatures.from_bytes(f.read())


def save_features(features_to_save: ExtractedFeatures, path: str) -> None:
    with open(path, mode="wb") as f:
        f.write(features_to_save.to_bytes())


def test_metrics(matcher: Matcher, model: Model, node_name: str) -> dict[str, float]:
    LOGGER.info("\n * * * \n * * * \n")
    LOGGER.info(f"Testing matcher {matcher.name} on model {model.name} tag {tag} node {node_name}")

    label_to_features: dict[int, list[ExtractedFeatures | str]] = defaultdict(list)
    if not HAVE_ENOUGH_RAM:
        FsTools.mkdir(f"{PathOrganizer.get_root()}/TEMP_EMBEDDINGS")
    for image, label in tqdm(testset, desc="Extracting features"):
        features = model.extract_features(node_name, image)
        if HAVE_ENOUGH_RAM:
            label_to_features[label].append(features)
        else:
            f_uuid = str(uuid.uuid4())
            f_path = f"{PathOrganizer.get_root()}/TEMP_EMBEDDINGS/{f_uuid}"
            save_features(features, f_path)
            label_to_features[label].append(f_path)

    # TESTING RECALL
    label_tps: dict[int, int] = defaultdict(int)
    label_fns: dict[int, int] = defaultdict(int)
    label_recall: dict[int, float] = {}
    for label, features_list in label_to_features.items():
        if len(features_list) > 1:
            if not HAVE_ENOUGH_RAM:
                features_list = [load_features(f_path) for f_path in features_list]
            for features1, features2 in itertools.combinations(features_list, 2):
                is_matched = matcher.match(features1, features2)
                label_tps[label] += is_matched
                label_fns[label] += not is_matched
            label_recall[label] = label_tps[label] / (label_tps[label] + label_fns[label])
            LOGGER.info(f"Recall for label {label} = {label_recall[label]}")
    total_tps = sum(label_tps.values())
    total_fns = sum(label_fns.values())
    total_recall = total_tps / (total_tps + total_fns)
    LOGGER.info(f"Total recall = {total_recall}")

    # TESTING FALSE POSITIVE RATE
    label_tns: dict[tuple[int, int], int] = defaultdict(int)
    label_fps: dict[tuple[int, int], int] = defaultdict(int)
    label_FPR: dict[tuple[int, int], float] = {}
    for (label1, l1_features), (label2, l2_features) in itertools.combinations(label_to_features.items(), 2):
        if len(l1_features) > 0 and len(l2_features) > 0:
            if not HAVE_ENOUGH_RAM:
                l1_features = [load_features(f_path) for f_path in l1_features]
                l2_features = [load_features(f_path) for f_path in l2_features]
            for features1, features2 in itertools.product(l1_features, l2_features):
                is_matched = matcher.match(features1, features2)
                label_tns[(label1, label2)] += not is_matched
                label_fps[(label1, label2)] += is_matched
            label_FPR[(label1, label2)] = (
                    label_fps[(label1, label2)] / (label_tns[(label1, label2)] + label_fps[(label1, label2)])
            )
            LOGGER.info(f"False Positive Rate for labels ({label1}, {label2}) = {label_FPR[(label1, label2)]}")
    total_tns = sum(label_tns.values())
    total_fps = sum(label_fps.values())
    total_fpr = total_fps / (total_fps + total_tns)
    LOGGER.info(f"Total False Positive Rate = {total_fpr}")
    LOGGER.info(f"Total recall = {total_recall}")

    LOGGER.info(f"Confusion matrix: \n"
                f"TPs = {total_tps}, FNs = {total_fns}, \n"
                f"FPs = {total_fps}, TNs = {total_tns}")
    total_accuracy = (total_tps + total_tns) / (total_tps + total_tns + total_fps + total_fns)
    LOGGER.info(f"Total Accuracy = {total_accuracy}")
    total_sensitivity = total_tps / (total_tps + total_fns)
    LOGGER.info(f"Total Sensitivity = {total_sensitivity}")
    total_specificity = total_tns / (total_tns + total_fps)
    LOGGER.info(f"Total Specificity = {total_specificity}")
    total_balanced_accuracy = (total_specificity + total_sensitivity) / 2
    LOGGER.info(f"Total Balanced Accuracy = {total_balanced_accuracy}")
    return {
        "balanced_accuracy": total_balanced_accuracy,
        "specificity": total_specificity,
        "sensitivity": total_sensitivity
    }


for matcher in MATCHERS:
    for model_name, tag, node_name in MODELS_TAGS_NODES:
        model = get_model_by_name(model_name)
        model.load_finetuned(tag)
        res = test_metrics(matcher, model, node_name)
