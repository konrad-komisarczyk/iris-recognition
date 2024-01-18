import torch

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.matchers.matcher import Matcher


class EuclideanDistanceMatcher(Matcher):
    """
    Class representing matcher. Used to match extracted features.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"EuclideanDistanceMatcher(threshold={self.threshold})"

    @staticmethod
    def distance(features_1: ExtractedFeatures, features_2: ExtractedFeatures) -> float:
        return float((features_1.flatten() - features_2.flatten()).pow(2).sum().sqrt())

    def match(self, features_1: ExtractedFeatures, features_2: ExtractedFeatures) -> bool:
        """
        :param features_1: features from 1 image
        :param features_2: features from 2 image
        :return: whether these are images of the same eye
        """
        if features_1.shape() != features_2.shape():
            raise ValueError(f"Features shapes do not match. {features_1.shape()} != {features_2.shape()}.")
        distance = EuclideanDistanceMatcher.distance(features_1, features_2)
        return distance <= self.threshold
