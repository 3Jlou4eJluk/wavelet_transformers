import numpy as np 
import torch
from torch import nn




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
