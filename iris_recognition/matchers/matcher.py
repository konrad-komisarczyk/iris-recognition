import abc

from iris_recognition.extracted_features import ExtractedFeatures


class Matcher(abc.ABC):
    """
    Class representing matcher. Used to match extracted features.
    """

    @abc.abstractmethod
    def match(self, features_1: ExtractedFeatures, features_2: ExtractedFeatures) -> bool:
        """
        :param features_1: features from 1 image
        :param features_2: features from 2 image
        :return: whether these are images of the same eye
        """
