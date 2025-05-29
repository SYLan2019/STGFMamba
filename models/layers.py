import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GCN(nn.Module):
    def __init__(self, in_steps, model_dim):
        super().__init__()
        self.model_dim = model_dim

        self.W1 = nn.Parameter(torch.randn(in_steps))  # (in_steps)
        self.W2 = nn.Parameter(torch.randn(model_dim, model_dim))  # (model_dim, model_dim)
    def forward(self, A, X):
        """
        A: (in_steps, num_nodes, num_nodes)
        X: (batch_size, in_steps, num_nodes, model_dim)
        """
        batch_size, in_steps, num_nodes, model_dim = X.shape

        XW1 = torch.einsum('btnd,t->btnd', X, self.W1)  # (batch_size, in_steps, num_nodes, model_dim)

        A_hat = self.normalize_adjacency(A)  # (in_steps, num_nodes, num_nodes)

        GCN_output = torch.einsum('tnm,btmd->btnd', A_hat, XW1)  # (batch_size, in_steps, num_nodes, model_dim)

        output = torch.einsum('btnd,dk->btnk', GCN_output, self.W2)  # (batch_size, in_steps, num_nodes, model_dim)

        output = F.softmax(output, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        return output

    def normalize_adjacency(self, A):
        """
        A: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = A.shape
        I = torch.eye(num_nodes).unsqueeze(0).to(A.device)  # (1, num_nodes, num_nodes)
        A_hat = A + I  # (batch_size, num_nodes, num_nodes)
        D = torch.sum(A_hat, dim=2)  # (batch_size, num_nodes)
        D_inv_sqrt = torch.diag_embed(D.pow(-0.5))  # (batch_size, num_nodes, num_nodes)
        A_hat = torch.bmm(D_inv_sqrt, torch.bmm(A_hat, D_inv_sqrt))  # (batch_size, num_nodes, num_nodes)
        return A_hat
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_real = nn.Parameter(torch.randn(out_features))
        self.bias_imag = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        real = F.linear(x.real, self.weight_real, self.bias_real) - F.linear(x.imag, self.weight_imag, self.bias_imag)
        imag = F.linear(x.real, self.weight_imag, self.bias_imag) + F.linear(x.imag, self.weight_real, self.bias_real)
        return torch.complex(real, imag)


class TemporalEmbedding(nn.Module):

    def __init__(self, in_dim: int,
                 emb_dim: int,
                 steps_per_day: int,
                 use_fan:bool=False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.steps_per_day = steps_per_day

        self.feat_embedding = nn.Linear(in_dim, emb_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, emb_dim)  # Timestamp-of-Day embedding
        self.dow_embedding = nn.Embedding(7, emb_dim)  # Day-of-Week embedding
        self.use_fan = use_fan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_steps, num_nodes, in_dim)
        # out: (batch_size, in_steps, num_nodes, 3*emb_dim) not use_fan

        feat = x[..., :self.in_dim]
        tod = x[..., 1]
        dow = x[..., 2]
        out = []

        # Feature Embedding
        feat_emb = self.feat_embedding(feat)
        out.append(feat_emb)
        # Time-of-Day Embedding
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
        out.append(tod_emb)

        # Day-of-Week Embedding
        dow_emb = self.dow_embedding(dow.long())
        out.append(dow_emb)
        # Concatenation
        out = torch.cat(out, dim=-1)

        return out


