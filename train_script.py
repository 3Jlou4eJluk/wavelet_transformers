
from train_script_args import argparser
args = argparser.parse_args()

import numpy as np
import torch
import tensorflow as tf
import pywt
import math
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from einops import rearrange
from einops.layers.torch import Rearrange
from einops import reduce
from einops import repeat
from torch import nn
from tqdm import tqdm

import torch
import math


from learning_process import Learner, ModelCheckpoint



transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])


transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])


training_dataset = datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
) # Data augmentation is only done on training images

validation_dataset = datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)


simple_train_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=args.batch_size,
    shuffle=False
)
simple_test_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=args.batch_size,
    shuffle=False
)


# @title SPT definition


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, merging_size=args.patch_size, exist_class_t=False, is_pe=False, wavelet='db1'):
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
            waves = torch.tensor(reduce(list(waves[1]), 'mat_no b c h w -> b c h w', 'mean')).to(device)

            out = self.merging(out)
            waves_embeddings = self.wavelet_merging(waves)
            out = out + waves_embeddings

        return out

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


# @title LSA
from einops import einsum
from torch.nn import Module

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), ** kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.mask is None:
            dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum(q, k, 'b h i d, b h j d -> b h i j'), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
        return x


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


class PatchEmbedding(nn.Module):
    def __init__(
            self, input_size, num_patches, embed_size,
            device=('cuda:0' if torch.cuda.is_available() else 'cpu'),
            enable_spt=False, additional_registers_count=0
    ):
        super().__init__()
        if enable_spt:
            self.linear_proj = ShiftedPatchTokenization(
                3, embed_size, args.patch_size, is_pe=True
            )
        else:
            self.linear_proj = nn.Linear(input_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patches * 2 + 1, embed_size))
        self.patch_no_embedding = nn.Embedding(
            num_patches + 1, embedding_dim=embed_size
        )

        self.registers_embeddings = nn.Embedding(
            additional_registers_count, embedding_dim=embed_size
        )

        self.num_patches = num_patches
        self.additional_registers_count = additional_registers_count
        self.device = device

    def forward(self, x):
        x = self.linear_proj(x)  # Преобразование патчей
        patch_no_tokens = torch.tensor([i for i in range(self.num_patches)], dtype=torch.int) + 1
        patch_no_tokens = patch_no_tokens.to(self.device)
        batch_size = x.shape[0]
        patch_no_tokens = repeat(
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
            last_bn_enable=True, output_size=None, norm_mode=None
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


### 4. Собираем всё вместе в ViT

class VisionTransformer(nn.Module):
    def __init__(
            self, input_size, num_patches, embed_size, num_classes, num_heads,
            num_layers, dropout, head_num_layers, head_dp, enable_spt
        ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            input_size, num_patches, embed_size, enable_spt=enable_spt
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

        self.apply(init_weights)

    def forward(self, patches):
        # преобразование патчей в эмбеддинг
        embeddings = self.patch_embedding(patches)
        # трансформер (само внимание)
        transformed = self.transformer(embeddings)
        # классификационный head, берём только [CLS] токен
        logits = self.head(transformed[:, 0])
        return logits


import sys
from learning_process import ExperimentResult

def finalize_experiment(learner_obj):
    if learner_obj is None:
        exp_res = ExperimentResult(
            -1, -1, -1, -1, -1, -1
        )
    else:
        with learner_obj.metrics as d:
            exp_res = ExperimentResult(
                d['train_loss_epoch_no'],
                d['val_loss_epoch_no'],
                d['train_acc'],
                d['val_acc'],
                d['train_loss'],
                d['val_loss']
            )
    exp_res.save()
    sys.exit(0)

try: 
    # @title Обучаем ViT
    model = VisionTransformer(
        48, (32 * 32) // (args.patch_size ** 2), 
        args.embedding_size, 10, 8, 8, 0., 4, 0.1, enable_spt=True
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, int(args.initial_period * len(simple_train_loader)), 
        int(args.period_increase_mult * len(simple_train_loader)), args.lr * args.min_lr
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    learner = Learner(
        model, optimizer, loss_fn, scheduler,
        simple_train_loader, simple_test_loader,
        device, args.n_epochs, checkpoint_path='/content/data/model_checkpoints',
        disable_checkpoints=True
    )

    learner.train()

    learner.metrics['val_acc']

except:
    finalize_experiment(None)

