from irisverify.settings import FEATURE_EXTRACTOR, MATCHER

from iris_recognition.extracted_features import ExtractedFeatures


def verify(iris_image_path: str, reference_features: ExtractedFeatures) -> bool:
    input_features = FEATURE_EXTRACTOR.extract_features(iris_image_path)
    return MATCHER.match(input_features, reference_features)
