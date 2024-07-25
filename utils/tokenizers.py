import numpy as np
import torch

from preprocessing_tools import build_batch_features


class WaveletTokenizer:
    def __init__(
            self, n_levels, coef_rate, 
            use_original_data, add_approx, 
            n_bins=5, ch_count=3, 
            return_tensor=False
    ):
        self.params = {
            'n_levels' : n_levels,
            'coef_rate' : coef_rate,
            'use_original_data' : use_original_data,
            'add_approx' : add_approx,
            'n_bins' : n_bins,
            'ch_count' : ch_count,
            'return_tensor' : return_tensor,
            'n_bins' : n_bins
        }

    def initialize(self):
        raise NotImplementedError

    def tokenize(self, images, labels):
        res = build_batch_features(
            images, labels, self.params['n_levels'],
            self.params['coef_rate'], self.params['ch_count'],
            self.params['use_original_data'], self.params['add_approx_data'],
            self.params['return_tensor'], self.params['n_bins']
        )

        return res
