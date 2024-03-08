import numpy as np 
import torch
import math
import ptwt
import pywt

from torch import nn
from einops import rearrange, reduce
from functools import partial


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        # Создаём матрицу ‘max_len’ на ‘d_model’, заполняем нулями.
        pe = torch.zeros(max_len, d_model)
        # Создаём вектор положений (одномерный).
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Уменьшаем влияние с увеличением частоты.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Применяем паттерны синусов и косинусов.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # Делаем ‘pe’ постоянной и не требующей градиентов.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Добавляем эмбеддинги, сберегая исходные размеры ‘x’.
        x = x + self.pe[:x.size(0), :]
        return x


class LinearModule(nn.Module):
    def __init__(
            self, input_size, hidden_size, n_layers, bn_enable, dp, 
            last_bn_enable=True, output_size=None
        ):
        super(LinearModule, self).__init__()

        self.layers = nn.ModuleList()
        prev_features = input_size

        for i in range(n_layers):
            self.layers.append(nn.Linear(prev_features, hidden_size))
            prev_features = hidden_size

            if bn_enable:
                if (i < n_layers - 1) or last_bn_enable:
                    self.layers.append(nn.BatchNorm1d(hidden_size))

            self.layers.append(nn.Dropout(dp))
            self.layers.append(nn.ReLU())

        if output_size is not None:
            self.layers.append(nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerModelV4(torch.nn.Module):
    def __init__(
            self, input_size, trans_hid_size, emb_dim, output_size, nhead, trans_nlayers, 
            trans_dp, coefs_discr_N, main_ff_hid_size, main_ff_nlayers, main_bn_enable, 
            main_dp, total_token_count
        ):

        super(TransformerModelV4, self).__init__()

        self.embedding = nn.Embedding(total_token_count, embedding_dim=emb_dim)
        self.positional_encoding = PositionalEncoding1D(emb_dim, max_len=input_size)
    
        encoder_layers = nn.TransformerEncoderLayer(
            emb_dim, 
            nhead, 
            trans_hid_size, 
            dropout=trans_dp,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            trans_nlayers
        )
        self.mid_bn = nn.Identity()

        if main_bn_enable:
            self.mid_bn = nn.BatchNorm1d(
                emb_dim * input_size
            )
        
        self.main_ff = LinearModule(
            emb_dim * input_size, 
            main_ff_hid_size,
            main_ff_nlayers,
            dp=main_dp,
            bn_enable=main_bn_enable
        )

        # Дополнительные слои для классификации
        self.fc_combined = nn.Linear(main_ff_hid_size, output_size)

    def forward(self, x, x_src_mask=None):
        o = self.embedding(x)
        o = self.transformer_encoder(o, x_src_mask)
        o = self.mid_bn(o.flatten(start_dim=1))
        o = self.main_ff(o)
        o = self.fc_combined(o)
        return o


class WaveLinear(nn.Module):
    def __init__(self, patch_size=8, image_shape=(32, 32), emb_dim=256, wave='bior2.2'):
        super(WaveLinear, self).__init__()
        self.linear_proj = nn.Linear(
            patch_size ** 2, emb_dim
        )
        self.wave_proj = nn.LazyLinear(
            emb_dim
        )
        self.color_proj1 = nn.Linear(
            patch_size ** 2, emb_dim
        )
        self.color_proj2 = nn.Linear(
            patch_size ** 2, emb_dim
        )

        # service fields
        self.patch_size = patch_size
        self.wavelet_transformation = partial(
            ptwt.wavedec2, wavelet=pywt.Wavelet(wave), level=1
        )


    def forward(self, x):
        patches = rearrange(
            x, f"bs c (h ph) (w pw) -> c bs (h w) ph pw",
            ph = self.patch_size, pw = self.patch_size
        )
        shape_patches, color_patches1, color_patches2 = patches[0], patches[1], patches[2]

        # shape of this thing is "bs (h * w) (ph // 2) (pw // 2)"
        wavelet_patches = self.wavelet_transformation(shape_patches)[1]

        patch_embeddings = self.linear_proj(
            rearrange(shape_patches, "bs pc ph pw -> bs pc (ph pw)")
        )
        wave_embeddings = self.wave_proj(
            rearrange(list(wavelet_patches), "mat_no bs pc ph pw -> bs pc (mat_no ph pw)")
        )
        color1_embeddings = self.color_proj1(
            rearrange(color_patches1, "bs pc ph pw -> bs pc (ph pw)")
        )
        color2_embeddings = self.color_proj2(
            rearrange(color_patches2, "bs pc ph pw -> bs pc (ph pw)")
        )

        result_embeddings = patch_embeddings + wave_embeddings \
                          + color1_embeddings + color2_embeddings
        return result_embeddings

