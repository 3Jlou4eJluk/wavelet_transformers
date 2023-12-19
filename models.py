import numpy as np 
import torch



from torch import nn
import torch

class LinearModule(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bn_enable, dp, last_bn_enable=True, output_size=None):
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


class GeneralTransformerV3(nn.Module):
    def __init__(self, 
                 input_size1, 
                 input_size2, 
                 d_hid, 
                 embedding_dim, 
                 output_size, 
                 nhead,
                 trans_nlayers,
                 trans_dp,
                 coefsubnet_nlayers,
                 coefsubnet_hidden_size,
                 coefsubnet_bn_enable,
                 coefsubnet_dp,
                 main_ff_hidden_size, 
                 main_ff_nlayers,
                 main_bn_enable,
                 main_dp,
                 total_token_count,
                 n_level=4
                 ):
        super(GeneralTransformerV3, self).__init__()
        
        self.n_levels = n_level
        self.input_size2 = input_size2

        # Создадим слой встраивания для первого входа
        self.emb_dim = embedding_dim
        self.embedding = nn.Embedding(total_token_count, embedding_dim=embedding_dim)
        
        # Делим с округлением вверх и умножаем обратно
        self.coef_n_tokens = (input_size2 + embedding_dim - 1) // embedding_dim
        actual_size = embedding_dim * self.coef_n_tokens
        self.coefnet = LinearModule(
            input_size2,
            coefsubnet_hidden_size,
            coefsubnet_nlayers,
            coefsubnet_bn_enable,
            coefsubnet_dp,
            output_size=actual_size
        )

        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, 
            nhead, 
            d_hid, 
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
                embedding_dim * (input_size1 + self.coef_n_tokens)
            )

        if main_ff_hidden_size <= 1:
            main_ff_hidden_size = int(np.ceil(main_ff_hidden_size * (embedding_dim * (input_size1 + self.coef_n_tokens))))

        self.main_ff = LinearModule(
            embedding_dim * (input_size1 + self.coef_n_tokens), 
            main_ff_hidden_size,
            main_ff_nlayers,
            dp=main_dp,
            bn_enable=main_bn_enable
        )

        # Дополнительные слои для обработки объединенных результатов
        self.fc_combined = nn.Linear(main_ff_hidden_size, output_size)

    def forward(self, x1, x2, x1_src_mask):
        # Проход через слои и объединение результатов
        o1 = self.embedding(x1)
        
        o2 = self.coefnet(x2)
        coef_net_out = []
        for i in range(self.coef_n_tokens):
            cur_slice = o2[:, i * self.emb_dim: (i + 1) * self.emb_dim]
            coef_net_out.append(cur_slice)
            
        coef_net_out = torch.stack(coef_net_out, dim=1)
        o1 = torch.cat((o1, coef_net_out), dim=1)
        
        o1 = self.transformer_encoder(o1, x1_src_mask)
        out = self.mid_bn(o1.flatten(start_dim=1))
        out = self.main_ff(out)
        out = self.fc_combined(out)
        return out



class TransformerModelV4(torch.nn.Module):
    def __init__(self,
                 input_size,
                 trans_hid_size,
                 emb_dim,
                 output_size,
                 nhead,
                 trans_nlayers,
                 trans_dp,
                 coefs_discr_N,
                 main_ff_hid_size,
                 main_ff_nlayers,
                 main_bn_enable,
                 main_dp,
                 total_token_count
                 ):
        super(TransformerModelV4, self).__init__()

        self.embedding = nn.Embedding(total_token_count, embedding_dim=emb_dim)
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
