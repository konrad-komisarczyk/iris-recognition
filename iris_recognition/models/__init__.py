from iris_recognition.models.alexnet import Alexnet
from iris_recognition.models.densenet import Densenet
from iris_recognition.models.googlenet import Googlenet
from iris_recognition.models.model import Model
from iris_recognition.models.resnet import Resnet
from iris_recognition.models.vgg import Vgg
from iris_recognition.models.vit import Vit


pretrained_model_name_to_class: dict[str, ] = {
    "resnet152": Resnet,
    "VGG16": Vgg,
    "DenseNet": Densenet,
    "AlexNet": Alexnet,
    "ViT": Vit,
    "GoogLeNet": Googlenet,
}


def get_pretrained_model_by_name(model_name: str) -> Model:
    """
    :param model_name: model name
    :return: PretrainedModel object
    """
    if model_name not in pretrained_model_name_to_class:
        raise ValueError(f"Unknown model {model_name}")
    return pretrained_model_name_to_class[model_name]()
