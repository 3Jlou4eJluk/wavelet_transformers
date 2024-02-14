import torch
import numpy as np
import pywt
import ptwt
import torchvision.transforms.functional as TF

from PIL import Image
from functools import partial
from einops import rearrange, reduce

class WaveletAugmentation:
    def __init__(self, levels, proba=0.5, complexity_range=(1, 16), wavelet='db1', verbose=False):
        self.wavelet = wavelet
        self.complexity_range = complexity_range
        self.levels = levels
        self.verbose = verbose
        self.proba = proba

        assert (proba >= 0.) and (proba <= 1.), 'Probability must be between 0. and 1.'
        self.wt_transform = partial(
            pywt.wavedec2, wavelet=pywt.Wavelet(wavelet),
            level=levels, mode='constant'
        )
        self.inverse_wt_transform = partial(
            pywt.waverec2, wavelet=pywt.Wavelet(wavelet)
        )

    def __call__(self, img):
        # Преобразование PIL изображения в тензор PyTorch
        img_tensor = TF.to_tensor(img)

        # Применяем аугментацию если выпал шанс
        if not np.random.binomial(1, self.proba, 1)[0]:
            # Возвращаем изображение в формате PIL для удобства
            return TF.to_pil_image(img_tensor)

        coefs = self.wt_transform(img_tensor.numpy())
        for i in range(self.levels):
            part = coefs[-1 - i]
            tens = rearrange(list(part), 'n c h w -> (n c) h w')
            abs_tens = torch.abs(tens)
            threshold = 2 ** (torch.log2(reduce(abs_tens, 'c h w -> c 1 1', 'max'))).to(torch.float)
            trade_off = torch.rand(1) * (self.complexity_range[1] - self.complexity_range[0]) \
                        + self.complexity_range[0]
            threshold /= trade_off
            mask = (abs_tens <= threshold)
            if self.verbose:
                print(f'Note: {mask.sum()} elements are zero now.')
            tens[mask] = 0
            back = rearrange(tens, '(n c) h w -> n c h w', c=3)
            coefs[-1 - i] = tuple([back[j] for j in range(3)])

        # Восстанавливаем изображение
        img_reconstructed = self.inverse_wt_transform(coefs)

        # Преобразуем обратно в тензор PyTorch для плавности операций
        img_reconstructed_tensor = torch.tensor(img_reconstructed, dtype=torch.float32)

        # Возвращаем изображение в формате PIL
        return TF.to_pil_image(img_reconstructed_tensor)


        
        
