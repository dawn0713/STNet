__all__ = ['STNet']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *

import datetime

# Cell
class STNet(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    def __init__(self, c_in: int, target_dim: int, patch_len: int, stride: int, num_patch: int,
                 n_layers: int = 3, d_model=128, n_heads=16, shared_embedding=True, d_ff: int = 256,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 res_attention: bool = True, pre_norm: bool = False, store_attn: bool = True,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 head_type="prediction", individual=False,
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression',
                             'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        self.backbone = STNetEncoder(c_in, num_patch=num_patch, patch_len=patch_len,
                                     n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                                     shared_embedding=shared_embedding, d_ff=d_ff,
                                     attn_dropout=attn_dropout, dropout=dropout, act=act,
                                     res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                     pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len,
                                     head_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)

    def forward(self, z):
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """
        z = self.backbone(z)  # z: [bs x nvars x d_model x num_patch]
        z = self.head(z)
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z


class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * num_patch]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)  # x: [bs x nvars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)  # x: [bs x nvars x forecast_len]
        return x.transpose(2, 1)  # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
        return x


class STNetEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        
        # attention bias
        self.nodevec1 = nn.Parameter(torch.randn(self.n_vars, 10), requires_grad=True)  # (num_nodes,10)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.n_vars), requires_grad=True)

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # Input encoding: projection the whole unvariate time series onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len * num_patch, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        self.gate_fusion = Gate_Fusion(d_model)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=False)

        self.iencoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                   store_attn=True)

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # iTransformer
        # self-attention between n_vars
        ix = torch.reshape(x, (bs, n_vars, num_patch * patch_len))  # ix: [bs x n_vars x num_patch * patch_len]
        ix = self.dropout(ix)
        ix = self.value_embedding(ix)  # ix: [bs x n_vars x d_model]

        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        # Encoder
        iz = self.iencoder(ix, adp) # iz: [bs x n_vars x d_model]
        iz = torch.reshape(iz, (bs, n_vars, 1, self.d_model))

        # channel-independence
        # self-attention between patches within the univariate time series
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)  # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1, 2)  # x: [bs x nvars x num_patch x d_model]

        u = torch.reshape(x, (bs * n_vars, num_patch, self.d_model))  # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x num_patch x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1, n_vars, num_patch, self.d_model))  # z: [bs x nvars x num_patch x d_model]
        
        # z = z + iz
        output = self.gate_fusion(iz, z)        
        output = output.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x num_patch]
        return output


class Gate_Fusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_s = nn.Linear(d_model, d_model)
        self.W_t = nn.Linear(d_model, d_model)

    def forward(self, x_s, x_t):
        hs = self.W_s(x_s)
        ht = self.W_t(x_t)
        g = torch.sigmoid(hs + ht)
        ones = torch.ones_like(g)
        z = g * hs + (ones-g) * ht
        return z


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=True):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) for i in
                                     range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, adp: Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, adp, prev=scores)
            return output
        else:
            for mod in self.layers:
                output = mod(output, adp)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=True,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout,
                                            res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, adp: Optional[Tensor]=None, prev: Optional[Tensor] = None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, adp, prev)
        else:
            src2, attn = self.self_attn(src, src, src, adp)
        if self.store_attn:
            self.attn = attn
            # current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            # file_name = f"tensor_{current_time}.pt"
            # torch.save(self.attn, file_name)
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
