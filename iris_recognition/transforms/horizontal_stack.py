import numpy as np
import torch
from PIL import Image
from torchvision.utils import _log_api_usage_once


class HorizontalStack(torch.nn.Module):
    """
    Horizontally stack the given image two times.
    """

    def __init__(self, p=2):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        imgs = [img for _ in range(self.p)]

        imgs_comb = np.vstack(imgs)
        imgs_comb = Image.fromarray(imgs_comb)

        return imgs_comb

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
