from __future__ import annotations

import torch
from PIL.Image import Image
from torch import nn
from torchvision.models import googlenet
from torchvision.models.feature_extraction import create_feature_extractor

from iris_recognition.extracted_features import ExtractedFeatures
from iris_recognition.models.model import Model


class Googlenet(Model):
    """
    Pretrained GoogLeNet model class
    """

    @property
    def name(self) -> str:
        return "GoogLeNet"

    def prepare_pretrained(self, num_classes: int) -> None:
        self.logger.debug("Loading model")
        self.model = googlenet(weights='DEFAULT')
        num_ftrs = self.model.fc.in_features
        # Replace the final fully connected layer (fc) with the new layers
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Final layer for num_classes classes
        )
        self.logger.debug("Done loading model and preparing classification layer")

    def extract_features(self, node_name: str, preprocessed_image: Image) -> ExtractedFeatures:
        """
        :param node_name: name of the node to extract features from
        :param preprocessed_image: preprocessed image
        :return: extracted features
        """
        feature_extractor = create_feature_extractor(self.model, [node_name])
        transform = self.get_transform()
        transformed_image = transform(preprocessed_image)
        if next(self.model.parameters()).is_cuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            transformed_image = transformed_image.to(device=device)

        extracted_features_value = feature_extractor(transformed_image.unsqueeze(0))[node_name]
        return ExtractedFeatures(extracted_features_value)
