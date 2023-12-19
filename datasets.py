import numpy as np
import torch

from preprocessing_tools import build_batch_features


class DatasetV4(torch.utils.data.Dataset):
    def __init__(self, images, labels, n_levels, coef_rate, 
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
        return self.objs_data[idx], self.labels[idx]
    
    def __len__(self):
        return self.objs_data.shape[0]

