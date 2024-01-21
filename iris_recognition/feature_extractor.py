from PIL import Image

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.models import get_model_by_name
from iris_recognition.preprocessing.preprocessor import Preprocessor


class FeatureExtractor:
    """
    Class for storing a model and using it to extract features from araw image
    """

    def __init__(self, model_name: str, model_tag: str, node_name: str) -> None:
        self.model = get_model_by_name(model_name)
        self.model.load_finetuned(model_tag)
        self.node_name = node_name
        self.preprocessor = Preprocessor()

    def extract_features(self, raw_image_path: str) -> ExtractedFeatures:
        """
        :param raw_image_path: raw image
        :return: extracted features
        """
        raw_image = Image.open(raw_image_path)
        preprocessed_image = self.preprocessor.preprocess_image(raw_image)
        return self.model.extract_features(self.node_name, preprocessed_image)
