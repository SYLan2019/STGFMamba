import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size, data):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        if data is not None:
            self.data = torch.nn.Parameter(torch.Tensor(data), requires_grad=True)
        else:
            self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]

class GCN(nn.Module):
    def __init__(self, in_steps, model_dim):
        super().__init__()
        self.model_dim = model_dim
        # 定义权重 W1 和 W2
        self.W1 = nn.Parameter(torch.randn(in_steps))  # (in_steps)
        self.W2 = nn.Parameter(torch.randn(model_dim, model_dim))  # (model_dim, model_dim)
        print('没使用GCN')
    def forward(self, A, X):
        """
        A: (in_steps, num_nodes, num_nodes)
        X: (batch_size, in_steps, num_nodes, model_dim)
        """
        batch_size, in_steps, num_nodes, model_dim = X.shape

        # 1. 计算 XW1: 对 in_steps 维度加权求和
        XW1 = torch.einsum('btnd,t->btnd', X, self.W1)  # (batch_size, in_steps, num_nodes, model_dim)

        # 2. 图卷积 GCN(A, XW1): 使用邻接矩阵 A 进行信息传播
        # 归一化邻接矩阵
        A_hat = self.normalize_adjacency(A)  # (in_steps, num_nodes, num_nodes)
        # 图卷积操作
        GCN_output = torch.einsum('tnm,btmd->btnd', A_hat, XW1)  # (batch_size, in_steps, num_nodes, model_dim)
#        GCN_output = torch.sigmoid(GCN_output)

        # 3. 线性变换: GCN_output · W2
        output = torch.einsum('btnd,dk->btnk', GCN_output, self.W2)  # (batch_size, in_steps, num_nodes, model_dim)

        # 4. Softmax: 对最后一个维度 (model_dim) 应用 Softmax
        output = F.softmax(output, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        return output

    def normalize_adjacency(self, A):
        """
        归一化邻接矩阵: A_hat = D^{-1/2} (A + I) D^{-1/2}
        A: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = A.shape
        # 添加自环
        I = torch.eye(num_nodes).unsqueeze(0).to(A.device)  # (1, num_nodes, num_nodes)
        A_hat = A + I  # (batch_size, num_nodes, num_nodes)
        # 计算度矩阵
        D = torch.sum(A_hat, dim=2)  # (batch_size, num_nodes)
        D_inv_sqrt = torch.diag_embed(D.pow(-0.5))  # (batch_size, num_nodes, num_nodes)
        # 归一化
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
        # x 是复数张量
        real = F.linear(x.real, self.weight_real, self.bias_real) - F.linear(x.imag, self.weight_imag, self.bias_imag)
        imag = F.linear(x.real, self.weight_imag, self.bias_imag) + F.linear(x.imag, self.weight_real, self.bias_real)
        return torch.complex(real, imag)
class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True).
            There is almost no difference between bias and non-bias in our experiments.
    """

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()

        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """

        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))

        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)

        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)

        return output


class TemporalEmbedding(nn.Module):

    def __init__(self, in_dim: int,
                 emb_dim: int,
                 steps_per_day: int,
                 use_fan:bool=False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.steps_per_day = steps_per_day

        self.feat_embedding = nn.Linear(in_dim, emb_dim) if not use_fan else FANLayer(input_dim=in_dim, output_dim=emb_dim)
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


class MambaBlock(nn.Module):

    def __init__(self, dim: int,
                 norm_cls=nn.LayerNorm,
                 layer_idx: Optional[int] = None,
                 bi_directional: bool = False) -> None:
        """
        From "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py"

        Simple block wrapping a Mamba with LayerNorm/RMSNorm and residual connection

        Here we have: Add -> LN -> Mamba, returning both
        the hidden_states (output of the Mamba) and the residual.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.dim = dim
        self.bi_directional = bi_directional

        self.norm = norm_cls(dim)
        self.mamba = Mamba(dim, layer_idx=layer_idx)  # TODO: Setting Hyperparameters
        if bi_directional:
            self.mamba_inverse = Mamba(dim, layer_idx=-layer_idx)
            self.linear = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
                inference_params=None):

        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mamba(LN(residual))
        """

        residual = (hidden_states + residual) if residual is not None else hidden_states
        h_norm = self.norm(residual.to(dtype=self.norm.weight.dtype))

        hidden_states = self.mamba(h_norm, inference_params=inference_params)
        if self.bi_directional:
            h_inverse = self.mamba_inverse(torch.flip(h_norm, [-2]),
                                           inference_params=inference_params)
            hidden_states = hidden_states + torch.flip(h_inverse, [-2])
            hidden_states = self.linear(hidden_states)

        return hidden_states, residual

#############################################################################
