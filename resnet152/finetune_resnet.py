from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
import torch
from torchvision.utils import _log_api_usage_once
from PIL import Image

# import pandas as pd
# import seaborn as sns
# from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
destination_dir = os.path.join(base_dir, "ubiris2_1_preprocessed")


class HorizontalStack(torch.nn.Module):
    """Horizontally stack the given image two times.
    """

    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        imgs = [img for _ in range(2)]

        imgs_comb = np.vstack(imgs)
        imgs_comb = Image.fromarray(imgs_comb)

        return imgs_comb

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


transform = transforms.Compose([
    HorizontalStack(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data = datasets.ImageFolder(destination_dir, transform=transform)

valid_size = 0.3
test_size = 0.5
batch_size = 1
num_workers = 0

indices = list(range(len(data)))
np.random.shuffle(indices)
split = int(np.floor(valid_size * len(data)))
train_idx, valid_idx = indices[split:], indices[:split]

valid_idx = valid_idx[int(np.floor(test_size * len(valid_idx))):]
test_idx = valid_idx[:int(np.floor(test_size * len(valid_idx)))]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# print(train_sampler.indices)
# print(valid_sampler)
# print(test_sampler)
# print(len(train_sampler))
# print(len(valid_sampler))
# print(len(test_sampler))


train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

# idx2class = {v: k for k, v in data.class_to_idx.items()}
#
#
# def get_class_distribution_loaders(dataloader_obj, dataset_obj):
#     count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
#
#     for _, j in dataloader_obj:
#         y_idx = j.item()
#         y_lbl = idx2class[y_idx]
#         count_dict[str(y_lbl)] += 1
#
#     return count_dict
#
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,7))
# sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution_loaders(train_loader, data)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Train Set')
# sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution_loaders(valid_loader, data)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Val Set')
# plt.show()
