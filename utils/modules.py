import numpy as np 
import torch
import math
import ptwt
import pywt

from torch import nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
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


# @title ViT definition

### 1. Embedding и Positional Encoding

import torch
import torch.nn as nn
from functools import partial
import einops as ein


def interleave_columns(arr1, arr2, device=('cuda:0' if torch.cuda.is_available() else 'cpu')):
    # Убедимся, что формы массивов совпадают
    if arr1.shape != arr2.shape:
        raise ValueError("Формы массивов должны быть одинаковым!")

    # Получаем размеры массивов
    rows, cols, depth = arr1.shape

    # Создаем массив, который будет хранить результат, теперь во второй оси
    arr1, arr2 = arr1.cpu(), arr2.cpu()
    result = torch.empty((rows, cols*2, depth), dtype=arr1.dtype)

    # Теперь интеркалируем вдоль второй оси
    for i in range(cols):
        result[:, 2*i, :] = arr1[:, i, :]
        result[:, 2*i + 1, :] = arr2[:, i, :]

    return result.to(device)


def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


### 2. Transformer Encoder

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


### 3. Classification Head

class LinearModule(nn.Module):
    def __init__(
            self, input_size, hidden_size, n_layers, dp,
            last_bn_enable=True, output_size=None, norm_mode='layer'
        ):
        super(LinearModule, self).__init__()
        global xavier_init

        norm_constructor = ((nn.BatchNorm1d) \
                            if norm_mode == 'batch' \
                            else (nn.LayerNorm))
        self.layers = nn.ModuleList()
        prev_features = input_size

        for i in range(n_layers):
            self.layers.append(nn.Linear(prev_features, hidden_size))
            xavier_init(self.layers[-1])
            prev_features = hidden_size

            if norm_mode is not None:
                if (i < n_layers - 1) or last_bn_enable:
                    self.layers.append(norm_constructor(hidden_size))

            self.layers.append(nn.Dropout(dp))
            self.layers.append(nn.ReLU())

        if output_size is not None:
            self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(
            self, embed_size, num_classes, hidden_size,
            n_layers, dp
        ):
        super().__init__()

        self.mlp_head = LinearModule(
            embed_size, hidden_size, n_layers,
            dp, last_bn_enable=False, output_size=num_classes,
            norm_mode='layer'
        )
    def forward(self, x):
        out = self.mlp_head(x)
        return out



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
        self.linear_proj = nn.Sequential(
                nn.LayerNorm(patch_size ** 2),
                nn.Linear(patch_size ** 2, emb_dim)
        )
        
        
        patch_size_act = (patch_size // 2 + 2 if wave == 'bior2.2' else patch_size // 2)
        self.wave_proj = nn.Sequential(
            nn.LayerNorm(patch_size_act ** 2),
            nn.Linear(patch_size_act ** 2) * 3, emb_dim
        )


        self.color_proj1 = nn.Sequential(
            nn.LayerNorm(patch_size ** 2),
            nn.Linear(patch_size ** 2, emb_dim)
        )

        self.color_proj2 = nn.Sequential(
            nn.LayerNorm(patch_size ** 2),
            nn.Linear(patch_size ** 2, emb_dim)
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


class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))

    def forward(self, x):

        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)

        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        #############################

        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        # #############################

        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1)
        #############################

        # out = self.out(x_cat)
        out = x_cat

        return out


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False, wavelet='db2', device='cpu'):
        super().__init__()

        self.wavelet = wavelet

        self.exist_class_t = exist_class_t

        self.patch_shifting = PatchShifting(merging_size)

        patch_dim = (in_dim*5) * (merging_size**2)
        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)

        self.is_pe = is_pe

        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )



        self.wavelet_merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size // 2, p2 = merging_size // 2),
            nn.LayerNorm(patch_dim // 4),
            nn.Linear(patch_dim // 4, dim)
        )

        self.device = device

    def forward(self, x):

        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)

        else:
            out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out = self.patch_shifting(out)
            waves = pywt.wavedec2(out.cpu(), wavelet=self.wavelet, level=1)
            waves = torch.tensor(reduce(list(waves[1]), 'mat_no b c h w -> b c h w', 'mean')).to(self.device)

            out = self.merging(out)
            waves_embeddings = self.wavelet_merging(waves)
            out = out + waves_embeddings

        return out


# @title Universal Embedding definition
class UniversalEmbedding(nn.Module):
    def __init__(
            self, image_shape, num_patches, embed_size,
            device='cpu',
            mode='vit', additional_registers_count=0,
            patch_size=8, wave='bior2.2'
    ):
        super().__init__()
        if mode == 'spt':
            self.linear_proj = ShiftedPatchTokenization(
                3, embed_size, patch_size, is_pe=True,
                device=device, wavelet=wave
            )
        elif mode == 'vit':
            self.linear_proj = nn.Linear(
                image_shape[0] * image_shape[1], embed_size
            )
        elif mode == 'wave_patches':
            self.linear_proj = WaveLinear(
                patch_size, image_shape, embed_size, wave
            )
        else:
            self.linear_proj = None


        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches * 2 + 1, embed_size))
        self.patch_no_embedding = nn.Embedding(
            num_patches + 1, embedding_dim=embed_size
        )
        '''
        self.registers_embeddings = nn.Embeddign(
            additional_registers_count, embedding_dim=embed_size
        )
        '''
        self.num_patches = num_patches
        self.additional_registers_count = additional_registers_count
        self.device = device


    def forward(self, x):
        x = self.linear_proj(x)  # Преобразование патчей
        patch_no_tokens = torch.tensor([i for i in range(self.num_patches)], dtype=torch.int) + 1
        patch_no_tokens = patch_no_tokens.to(self.device)
        batch_size = x.shape[0]
        patch_no_tokens = ein.repeat(
            patch_no_tokens[None, :], f'f pc -> ({batch_size} f) pc'
        )
        #cls_token = torch.zeros(1, 1, x.size(2)).int()  # Классифицирующий токен
        cls_token = torch.zeros((1, ), dtype=torch.int)
        cls_tokens = cls_token.expand(batch_size, -1).to(self.device)

        patch_no_tokens = torch.cat((cls_tokens, patch_no_tokens), dim=1)

        patch_no_embs = self.patch_no_embedding(patch_no_tokens)


        batch_size, _, _ = x.shape
        transformer_features = interleave_columns(patch_no_embs[:, 1:], x)

        cls_tokens = patch_no_embs[:, 0]
        cls_tokens = cls_tokens.unsqueeze(1)

        transformer_features = torch.cat((cls_tokens, transformer_features), dim=1)

        if not self.additional_registers_count:
            pass

        transformer_features += self.positional_encoding  # Добавление позиционного кодирования
        return transformer_features


# @title UniversalTransformer definition

class UniversalTransformer(nn.Module):
    def __init__(
            self, input_size, embed_size, num_classes, num_heads,
            num_layers, dropout, head_num_layers, head_dp,
            enable_lsa, device, mlp_dim_ratio=2, trans_d_hid=2048,
            patch_size=8, wave='bior2.2', add_reg=0, mode='wave_patches'
        ):
        super().__init__()
        num_patches = (32 // patch_size) ** 2
        self.patch_embedding = UniversalEmbedding(
            input_size, num_patches, embed_size, mode=mode,
            patch_size=patch_size, wave=wave, additional_registers_count=add_reg,
            device=device
        )

        self.transformer = TransformerEncoder(embed_size, num_heads, num_layers, dropout)

        # xavier init
        for layer in self.transformer.encoder.layers:
            # Инициализируем веса для первого linear слоя в каждом TransformerEncoderLayer
            torch.nn.init.xavier_uniform_(layer.linear1.weight)
            if layer.linear1.bias is not None:
                torch.nn.init.constant_(layer.linear1.bias, 0)
            # Тоже самое для второго linear слоя
            torch.nn.init.xavier_uniform_(layer.linear2.weight)
            if layer.linear2.bias is not None:
                torch.nn.init.constant_(layer.linear2.bias, 0)

        self.head = ClassificationHead(
            embed_size, num_classes, embed_size,
            head_num_layers, head_dp
        )

        self.apply(xavier_init)

    def forward(self, patches):
        # преобразование патчей в эмбеддинг
        embeddings = self.patch_embedding(patches)
        # трансформер (само внимание)
        transformed = self.transformer(embeddings)
        # классификационный head, берём только [CLS] токен
        logits = self.head(transformed[:, 0])
        return logits

