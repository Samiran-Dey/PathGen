from collections import OrderedDict
from os.path import join
import math
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_period=1000):
        super().__init__()
        self.dim = dim
        self.max_period=max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super(TimeEmbedding, self).__init__()
        self.dim = dim  
        half_dim = dim // 2 
        emb = np.log(max_period) / (half_dim - 1)  
        emb = np.exp(np.arange(half_dim) * -emb)
        self.emb = torch.tensor(emb, dtype=torch.float32).to(device)

    def forward(self, t):
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(device)


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            Mish(),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


