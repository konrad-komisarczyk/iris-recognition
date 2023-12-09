import abc

from PIL.Image import Image

from iris_recognition.extracted_features import ExtractedFeatures


class FinetunedModel(abc.ABC):
    """
    class representing model that we use to extract features
    """

    @abc.abstractmethod
    @property
    def name(self) -> str:
        """
        :return: name of the model
        """

    @abc.abstractmethod
    def load(self, tag: str) -> None:
        """
        Loads model with given tag from corresponding path
        :param tag: tag
        """

    @abc.abstractmethod
    def save(self, tag: str) -> None:
        """
        Saves model with given tag to corresponding path
        :param tag: tag
        """

    @abc.abstractmethod
    def extract_features(self, preprocessed_image: Image) -> ExtractedFeatures:
        """
        :param preprocessed_image: preprocessed image
        :return: extracted features
        """
