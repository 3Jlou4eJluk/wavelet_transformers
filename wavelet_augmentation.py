import torch
from PIL import Image
import numpy as np
import pywt
import ptwt
from functools import partial
from einops import rearrange
from einops import reduce
from einops import repeat

class WaveletAugmentation:
    def __init__(self, levels, trade_off=4, wavelet='db1', verbose=False):
        self.wavelet = wavelet
        self.trade_off = trade_off
        self.levels = levels
        self.verbose = verbose
        self.wt_transform = partial(
            ptwt.wavedec2, wavelet=pywt.Wavelet(wavelet), 
            level=levels, mode='constant'
        )
        self.inverse_wt_transform = partial(
            ptwt.waverec2, wavelet=pywt.Wavelet(wavelet)
        )
    def __call__(self, img):
        coefs = self.wt_transform(img)

        for i in range(self.levels):
            part = coefs[-1 - i]
            tens = rearrange(list(part), 'n c h w -> (n c) h w')
            abs = torch.abs(tens)
            thresholds = reduce(abs / self.trade_off, 'c h w -> c 1 1', 'max')
            mask = (abs <= thresholds)
            if self.verbose:
                print(f'Note: {mask.sum()} elements are zero now')
            tens[mask] = 0
            back = rearrange(tens, '(n c) h w -> n c h w', c=3)
            coefs[-1 - i] = tuple([back[j] for j in range(3)])
        # reconstructing image
        return self.inverse_wt_transform(coefs)
        
        