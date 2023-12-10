from iris_recognition.pretrained_models.alexnet import AlexnetPretrained
from iris_recognition.pretrained_models.densenet import DensenetPretrained
from iris_recognition.pretrained_models.googlenet import GooglenetPretrained
from iris_recognition.pretrained_models.pretrained_model import PretrainedModel
from iris_recognition.pretrained_models.resnet import ResnetPretrained
from iris_recognition.pretrained_models.vgg import VggPretrained
from iris_recognition.pretrained_models.vit import VitPretrained


pretrained_model_name_to_class: dict[str, ] = {
    "resnet152": ResnetPretrained,
    "VGG16": VggPretrained,
    "DenseNet": DensenetPretrained,
    "AlexNet": AlexnetPretrained,
    "ViT": VitPretrained,
    "GoogLeNet": GooglenetPretrained,
}


def get_pretrained_model_by_name(model_name: str) -> PretrainedModel:
    """
    :param model_name: model name
    :return: PretrainedModel object
    """
    if model_name not in pretrained_model_name_to_class:
        raise ValueError(f"Unknown model {model_name}")
    return pretrained_model_name_to_class[model_name]()
