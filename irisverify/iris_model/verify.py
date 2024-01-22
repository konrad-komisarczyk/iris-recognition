from irisverify.settings import FEATURE_EXTRACTOR, MATCHER

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.tools.logger import get_logger

LOGGER = get_logger("irisverify verify")


def verify(iris_image_path: str, reference_features: ExtractedFeatures) -> bool:
    LOGGER.info(f"Verifying user image: {iris_image_path}")
    input_features = FEATURE_EXTRACTOR.extract_features(iris_image_path)
    return MATCHER.match(input_features, reference_features, LOGGER)


def extract_feature_vector(iris_image_path: str) -> ExtractedFeatures:
    return FEATURE_EXTRACTOR.extract_features(iris_image_path)
