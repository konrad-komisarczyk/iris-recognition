from __future__ import annotations

import io

import torch
from torch import tensor, flatten


class ExtractedFeatures:
    """
    class for storing features extracted from the iris image
    """
    def __init__(self, data: tensor) -> None:
        self.data = data

    def shape(self) -> list[int]:
        """
        :return: shape of the features tensor
        """
        return [dim for dim in self.data.shape]

    def flatten(self) -> tensor:
        """
        :return: 1D tensor
        """
        return flatten(self.data)

    def to_bytes(self) -> bytes:
        """
        :return: bytes representation
        """
        buffer = io.BytesIO()
        torch.save(self.data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(bytes_data: bytes) -> ExtractedFeatures:
        """
        :param bytes_data: Extracted Features bytes representation
        :return: Extracted Features object
        """
        buffer = io.BytesIO(bytes_data)
        tensor_data = torch.load(buffer)
        return ExtractedFeatures(tensor_data)
