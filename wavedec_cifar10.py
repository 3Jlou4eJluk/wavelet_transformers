import numpy as np
import torch
import einops
import torch.nn as nn
import pickle
import time
import torchvision
import ptwt
import os


from functools import partial
import einops as ein

from tqdm import tqdm
from torchvision import transforms

# parameters
batch_size = 64
n_epochs = 300

class MLP_Atom(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.layer(x)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, layer_count):
        prev_out = input_dim
        self.layers = torch.nn.ParameterList(
            [MLP_Atom(prev_out, hidden_dim) for i in range(layer_count)] +
            [torch.nn.Linear(hidden_dim, out_dim)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ReprLearnEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=256, patch_size=8, in_channels=3, repr_channels=5):
        super(ReprLearnEmbedding, self).__init__()

        self.repr_layer = torch.nn.Linear(in_channels, repr_channels)
        self.projections = torch.nn.ParameterList(
            [torch.nn.Linear(patch_size ** 2, embedding_dim) for i in range(repr_channels)]
        )

        ###### service fields
        self.repr_channels = repr_channels
        self.patch_size = patch_size
    
    def forward(self, x):
        '''
            Expecting x.shape == (bs, c, h, w)
        '''
        new_channels = einops.rearrange(
            self.repr_layer(einops.rearrange(x, "bs c h w -> bs h w c")),
            "bs (h ph) (w pw) c -> c bs (h w) (ph pw)",
            ph=self.patch_size, pw=self.patch_size
        )

        repr = None
        for i in range(self.repr_channels):
            channel = new_channels[i]
            repr = (
                self.projections[i](channel) 
                    if repr is None 
                        else repr + self.projections[i](channel)
            )
        
        return repr


class WaveDecEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=256, patch_size=8, level=3):
        super(WaveDecEmbedding, self).__init__()

        wavedec_input_dim = sum([(patch_size // (2 ** lvl)) ** 2 for lvl in range(1, level + 1)])

        self.wavedec_projection = torch.nn.Linear(
            3 * wavedec_input_dim,
            out_features=embedding_dim
        )

        self.color_projection = torch.nn.Linear(
            (patch_size ** 2) * 2, embedding_dim
        )
        self.approx_projection = torch.nn.Linear(
            (patch_size // (2 ** level)), embedding_dim
        )

        self.wavedec = partial(ptwt.wavedec2, wavelet='haar', level=level)

        ###### service fields
        self.level = level
        self.patch_size = patch_size
    
    def forward(self, x):
        '''
            Expecting x.shape == (bs, c, h, w)
        '''
        yc = x[:, 0, :, :]
        color = x[:, 1:, :, :]

        color_patches = einops.rearrange(
            color, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph = self.patch_size,
            pw = self.patch_size
        )
    
        waves = self.wavedec(yc)
        approx = waves[0]
        approx_patches = einops.rearrange(
            approx, "b (h ph) (w pw) -> b (h w) (ph pw)", 
            ph = self.patch_size // (2 ** self.level),
            pw = self.patch_size // (2 ** self.level)
        )
        decomp_features = None
        for lvl in range(1, self.level + 1):
            decomp = waves[self.level - lvl + 1]
            decomp = einops.rearrange(
                list(decomp), "mat_no b (h ph) (w pw) -> b (h w) (mat_no ph pw)",
                ph = self.patch_size // (2 ** lvl),
                pw = self.patch_size // (2 ** lvl)
            )
            decomp_features = (
                decomp 
                    if decomp_features is None 
                    else torch.cat((decomp_features, decomp), dim=-1)
            )
        
        decomp_projected = self.wavedec_projection(decomp_features)
        approx_projected = self.approx_projection(approx_patches)
        color_projected = self.color_projection(color_patches)

        return decomp_projected + approx_projected + color_projected


# @title ViT definition

### 1. Embedding и Positional Encoding


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



# @title Universal Embedding definition
class TokenCollector(nn.Module):
    def __init__(
            self, image_shape, num_patches, embed_size,
            device='cpu', mode='vit', additional_registers_count=0,
            patch_size=8, wave='db1', wavedec_levels=3,
            nonlinear_embedding_layers_count=5, nonlinear_embedding_hidden_dim=256,
            repr_channels=5
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
            self.linear_proj = WaveEmbedding(
                embed_size,
                patch_size,
                wave,
                wavedec_levels,
                nonlinear_embedding_layers_count,
                nonlinear_embedding_hidden_dim
            )
        elif mode == 'channels_learning':
            self.linear_proj = ReprLearnEmbedding(
                embed_size, patch_size, repr_channels=repr_channels
            )
        elif mode == 'wavedec_embedding':
            self.linear_proj = WaveDecEmbedding(
                embed_size, patch_size, wavedec_levels
            )


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
            patch_size=8, wave='bior2.2', add_reg=0, mode='wave_patches',
            wavedec_levels=3, nonlinear_embedding_hidden_dim=256, 
            nonlinear_embedding_layers_count=5, repr_channels=5
        ):
        super().__init__()
        num_patches = (32 // patch_size) ** 2
        self.patch_embedding = TokenCollector(
            input_size, num_patches, embed_size, mode=mode,
            patch_size=patch_size, wave=wave, additional_registers_count=add_reg,
            device=device, wavedec_levels=wavedec_levels,
            nonlinear_embedding_layers_count=nonlinear_embedding_layers_count,
            nonlinear_embedding_hidden_dim=nonlinear_embedding_hidden_dim,
            repr_channels=repr_channels

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

# @title Learning tools


class EarlyStopping:
    """Ранняя остановка для остановки обучения, когда ошибка валидации перестает уменьшаться."""
    def __init__(self, patience=7, verbose=False, delta=0, disable=False):
        """
        Args:
            patience (int): Количество эпох без улучшения после которых обучение будет прекращено.
            verbose (bool): Включает вывод сообщений о ранней остановке.
            delta (float): Минимальное изменение между эпохами для рассмотрения как улучшение.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.disable = disable

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.disable:
            return None

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} из {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        if self.early_stop and self.verbose:
            print("Ранняя остановка выполнена")


class ModelContainer:
    def __init__(self, model):
        self.model = model
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_milestone": [],
            "val_milestone": []
        }

    def forward(self, x):
        return self.model.forward(x)

class Learner:
    def __init__(
            self, model, optimizer, loss_fn, scheduler,
            train_dl, val_dl, device, epochs, checkpoint_path=None,
            max_training_time=None, chill_time=120, early_stop=None,
            verbose=True
        ):

        self.model = model
        self.start_epoch = 0
        if isinstance(model, ModelContainer):
            self.model_container = model
            self.model = self.model_container.model
            if model.metrics['train_milestone']:
                self.start_epoch = max(model.metrics['train_milestone']) + 1
            print('Current start epoch is ', self.start_epoch)
        else:
            print('Pure model in args. Wrapping.')
            self.model_container = ModelContainer(model)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.max_training_time = max_training_time
        self.chill_time = chill_time
        self.early_stop = early_stop

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.epochs = epochs
        self.device = device

        self.verbose = verbose

        # service fields
        self.current_work_time = 0

    def train_epoch(self, log_train_quality=False, verbose=False, epoch_no=None):
        self.model.train()

        loss_sum = []
        acc_sum = []
        for x, y in tqdm(self.train_dl, disable=not verbose, leave=False, desc='Batch', position=1):
            start_time = time.time()
            if 'cpu' not in self.device:
                x = x.to(self.device)
                y = y.to(self.device)

            pred = self.model.forward(x)
            loss = self.loss_fn(pred, y.squeeze())
            loss.backward()

            pred_classes = torch.argmax(pred, dim=1)
            loss_sum.append(float(loss.item()))
            acc_sum.append(float((
                pred_classes.detach().squeeze() == y.squeeze()
            ).sum() / len(y)))

            self.optimizer.first_step(zero_grad=True)
            self.loss_fn(self.model(x), y.squeeze()).backward()  # make sure to do a full forward pass
            self.optimizer.second_step(zero_grad=True)

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()
            self.current_work_time += time.time() - start_time
            if (self.max_training_time is not None) and (current_work_time >= self.max_training_time):
                current_work_time = 0
                time.sleep(self.chill_time)

        loss_sum_val = sum(loss_sum) / len(self.train_dl)
        acc_sum_val = sum(acc_sum) / len(self.train_dl)

        if log_train_quality and (epoch_no is not None):
            self.model_container.metrics['train_milestone'].append(epoch_no)
            self.model_container.metrics['train_loss'].append(loss_sum_val)
            self.model_container.metrics['train_acc'].append(acc_sum_val)


    def validation_epoch(self, log_train_quality=False, verbose=False, epoch_no=None):
        self.model.eval()

        train_loss = 0.
        val_loss = 0.
        train_acc = 0.
        val_acc = 0.
        with torch.no_grad():
            if log_train_quality:
                for x, y in tqdm(self.train_dl, disable=not verbose, leave=False, desc='Validation: Train', position=2):
                    if 'cpu' not in self.device:
                        x = x.to(self.device)
                        y = y.to(self.device)
                    pred = self.model.forward(x)
                    pred_classes = torch.argmax(pred, dim=1)
                    train_acc += float((
                        pred_classes.detach().squeeze() == y.squeeze()
                    ).sum() / len(y))

                    loss = self.loss_fn(pred, y.squeeze())
                    train_loss += float(loss.item())
                train_loss /= len(self.train_dl)
                train_acc /= len(self.train_dl)

            for x, y in tqdm(self.val_dl, disable=not verbose, leave=False, desc='Validation: Test', position=2):
                if 'cpu' not in self.device:
                    x = x.to(self.device)
                    y = y.to(self.device)
                pred = self.model.forward(x)
                pred_classes = torch.argmax(pred, dim=1)
                val_acc += float((
                    pred_classes.detach().squeeze() == y.squeeze()
                ).sum() / len(y))

                loss = self.loss_fn(pred, y.squeeze())
                val_loss += float(loss.item())

            val_acc /= len(self.val_dl)
            val_loss /= len(self.val_dl)

            self.early_stop(val_loss, self.model)
            if self.early_stop.early_stop:
                # прекращаем обучение
                return None

        if epoch_no is not None:
            if log_train_quality:
                self.model_container.metrics['train_milestone'].append(epoch_no)
                self.model_container.metrics['train_loss'].append(train_loss)
                self.model_container.metrics['train_acc'].append(train_acc)

            self.model_container.metrics['val_milestone'].append(epoch_no)
            self.model_container.metrics['val_loss'].append(val_loss)
            self.model_container.metrics['val_acc'].append(val_acc)
            if len(self.model_container.metrics['val_acc']) == 1:
                print('Dumping best_model')
                with open(f'{self.checkpoint_path + "/best_container"}', 'wb') as f:
                    pickle.dump(self.model_container, f)
            else:
                if max(self.model_container.metrics['val_acc'][:-1]) < val_acc:
                    with open(f'{self.checkpoint_path + "/best_container"}', 'wb') as f:
                        pickle.dump(self.model_container, f)

    def train_cycle(self):
        for epoch in (pbar := tqdm(range(self.start_epoch, self.start_epoch + self.epochs), total=self.epochs, disable=False, desc='Epoch', position=0)):
            self.train_epoch(log_train_quality=True, verbose=self.verbose, epoch_no=epoch)
            if not epoch % 1:
                self.validation_epoch(log_train_quality=False, verbose=self.verbose, epoch_no=epoch)
                pbar.set_description(('Loss (Train/Test): {0:.3f}/{1:.3f}.\n' +\
                                     'Accuracy,% (Train/Test): {2:.2f}/{3:.2f}.\n' +\
                                     'On epoch_no: {4}').format(
                    self.model_container.metrics['train_loss'][-1], self.model_container.metrics['val_loss'][-1],
                    self.model_container.metrics['train_acc'][-1], self.model_container.metrics['val_acc'][-1],
                    epoch
                ))

    def train(self):
        if 'cpu' not in self.device:
            self.model.to(self.device)
        self.train_cycle()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



brute_transform = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      # WaveletAugmentation(levels=5),
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      #YCbCrTransform(),
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      #transforms.Normalize((0.4676, 0.4783, 0.5069), (0.0538, 0.0032, 0.0026))
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])


soft_transform = transforms.Compose([transforms.Resize((32,32)),
                            #YCbCrTransform(),
                               transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])


# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=brute_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=soft_transform)

# Define the data loaders for the train and test datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

if not os.path.isfile('checkpoints/best_container'):
    model = UniversalTransformer(
        (32, 32), 384, 10, 16, 16, 0.1, 8, 0.1,
        mode='wavedec_embedding', enable_lsa=True,
        patch_size=8, device=device, wave='db1',
        wavedec_levels=3, nonlinear_embedding_hidden_dim=384,
        nonlinear_embedding_layers_count=3, repr_channels=5
    )
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(), base_optimizer, lr=2e-5
    )
    # good parameters: cycle = 6, eta_min=5e-6. Optimizer lr = 2e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 6, eta_min=5e-6, T_mult=1
    )
    scheduler = None

    model.to(device)
else:
    print('Found best container, loading.')
    with open('checkpoints/best_container', 'rb') as f:
        model = pickle.load(f)
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.model.parameters(), base_optimizer, lr=4e-6
    )
    # good parameters: cycle = 6, eta_min=5e-6. Optimizer lr = 2e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 6, eta_min=8e-7, T_mult=1
    )
    scheduler = None

    model.model.to(device)


"""scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-6, total_steps=n_epochs*len(simple_train_loader),
    pct_start=0.1, final_div_factor=10, three_phase=True
)"""


loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

early_stop = EarlyStopping(patience=10, delta=0.01, disable=True)

learner = Learner(
    model, optimizer, loss_fn, scheduler,
    trainloader, testloader,
    device, n_epochs, checkpoint_path='checkpoints',
    early_stop=early_stop, verbose=True
)

learner.train()

