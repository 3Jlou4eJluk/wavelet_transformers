import numpy as np
import torch

from preprocessing_tools import build_batch_features
from torchvision.datasets import ImageFolder
from einops import reduce, rearrange

class DatasetV4(torch.utils.data.Dataset):
    def __init__(
            self, images, labels, n_levels, coef_rate, 
            use_original_data, add_approx, n_bins
        ):
        super(DatasetV4, self).__init__()
        self.objs_data, self.labels, self.tokens_per_obj = build_batch_features(
            images,
            labels,
            n_levels,
            coef_rate,
            use_original_data=use_original_data,
            add_approx=add_approx,
            n_bins = n_bins
        )
    
    def __getitem__(self, idx):
        return torch.tensor(self.objs_data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def __len__(self):
        return self.objs_data.shape[0]


class CombinedImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dirs, transform=None):
        self.datasets = [ImageFolder(root=root_dir, transform=transform) for root_dir in root_dirs]
        self.transform = transform

    def __getitem__(self, index):
        dataset_idx = index // 90000
        sample_idx = index % 90000
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)


class YCbCrTransform:
    def __init__(self):
        pass
    def __call__(self, pil_image):
        ycbcr_image = pil_image.convert('YCbCr')
        return ycbcr_image


class SimpleTransform:
    def __init__(self):
        pass
    def __call__(self, tensor_image):
        minn = reduce(tensor_image, 'c h w -> c 1 1', 'min')
        maxx = reduce(tensor_image, 'c h w -> c 1 1', 'max')
        tensor_image = (tensor_image - minn) / (maxx - minn)
        return tensor_image
