import numpy as np
import torch
from PIL import Image


class HorizontalStack(torch.nn.Module):
    """
    Horizontally stack the given image two times.
    """

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, img):
        imgs = [img for _ in range(self.p)]

        imgs_comb = np.vstack(imgs)
        imgs_comb = Image.fromarray(imgs_comb)

        return imgs_comb

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
