from idlelib.pyparse import trans

import torch
import torch.nn as nn

from typing import Tuple, Optional, Any, Union
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

from models.modules import CMambaEncoder, GraphPropagate
from models.layers import  TemporalEmbedding, MambaBlock, RecurrentCycle, GCN

P_BFS = 0.5  # < 1
Q_BFS = 2.  # > 1
P_DFS = 2.  # > 1
Q_DFS = 0.5  # < 1
P_RW = 1.
Q_RW = 1.


class STGCMamba(nn.Module):

    def __init__(self, in_dim: int,
                 out_dim: int,
                 emb_dim: int,
                 ff_dim: int,
                 steps_per_day: int,
                 num_nodes:int,
                 # graph: dgl.DGLGraph,
                 e_layers: int = 3,
                 d_model: int = 128,
                 gddmlp: bool = True,
                 # TODO:d_ff与adaptive_embedding_dim同步修改
                 d_ff: int = 128,
                 c_out: int = 12,
                 dt_rank: int = 32,
                 reduction: int = 2,
                 avg: bool = False,
                 bias: bool = True,
                 max: bool = False,
                 dt_init: str = 'random',
                 d_state: int = 16,
                 dt_max: float = 0.1,
                 dt_min: float = 0.001,
                 dt_init_floor: float = 1e-4,
                 dt_scale: float = 1.0,
                 adaptive_embedding_dim=32,
                 in_steps=12,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 order=2,
                 kernel_size=[1]) -> None:

        super().__init__()
        print(f'd_ff: {d_ff},adaptive_embedding_dim: {adaptive_embedding_dim}')
        # Model Configurations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.feat_embedding = nn.Linear(in_dim, emb_dim)
        self.temp_emb = TemporalEmbedding(in_dim, emb_dim, steps_per_day)
        # self.graph = graph
        # TODO:修改输入维数要记得修改这里
        emb_scan_dim = 3 * emb_dim + adaptive_embedding_dim
        # self.temporal_proj = TCNLayer(self.model_dim, self.model_dim, max_seq_length=in_steps)
        self.locals = GraphPropagate(Ks=order)
        self.emb_scan_dim = emb_scan_dim
        self.temporal_proj = nn.Conv2d(
            self.emb_scan_dim, self.emb_scan_dim, (1, kernel_size[0]), 1, 0
        )
        self.adaptive_embedding_dim=adaptive_embedding_dim
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        self.mamba_blocks = nn.ModuleList([
            CMambaEncoder(d_model = emb_scan_dim,
                          e_layers = e_layers,
                          d_ff = d_ff,
                          bias = bias,
                          gddmlp = gddmlp,
                          dt_rank = dt_rank,
                          d_state = d_state,
                          dt_init = dt_init,
                          c_out = c_out,
                          reduction = reduction,
                          avg = avg,
                          max = max,
                          dropout = dropout,
                          dt_max = dt_max,
                          dt_min = dt_min,
                          dt_init_floor = dt_init_floor,
                          dt_scale = dt_scale)
            for _ in range(order)])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_scan_dim, nhead=4, dim_feedforward=ff_dim, dropout=dropout),
            num_layers
        )
        self.gcn_layer = GCN(in_steps, emb_scan_dim)
        # Spatio-Temporal Projection
        self.ff_scan = nn.Sequential(
            nn.Linear(emb_scan_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # Regression Layer
        self.regression_layer = nn.Sequential(
            nn.Linear(self.emb_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, self.out_dim)
        )

        self.activation = nn.GELU()
        self.pws = nn.ModuleList(
            [nn.Linear(emb_scan_dim, emb_scan_dim) for _ in range(order)]
        )
        self.scale = [1, 0.01, 0.001]
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size[0]), stride=1)
        print(f'{order} prop+freq graph mamba+norm')
        print(f'e_layers:{e_layers}')
    def forward(self, x: torch.Tensor, out_steps: int = 12) -> Tuple[Tensor, Tensor, Tensor]:
        # 原始序列直接作为输入
        device = x.device
        batch_size, num_steps, num_nodes, _ = x.shape
        features = torch.tensor([]).to(x)
        # Input Processing
        temp_emb = self.temp_emb(x)
        if self.adaptive_embedding_dim > 0: # X_ste
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )# (batch_size, in_steps, num_nodes, adaptive_embedding_dim)
            features = torch.concat([features, self.dropout(adp_emb)], -1)
        input_emb = torch.cat(
            [temp_emb] + [features], dim=-1
        ) # (batch_size, in_steps, num_nodes, model_dim)
        input_emb = self.temporal_proj(input_emb.transpose(1, 3)).transpose(1, 3)
        graph = torch.matmul(self.adaptive_embedding, self.adaptive_embedding.transpose(1, 2))
        graph = self.pooling(graph.transpose(0, 2)).transpose(0, 2)
        graph = F.softmax(F.relu(graph), dim=-1)
        # graph(in_steps, num_nodes, num_nodes)
        x_loc = self.locals(input_emb, graph)
        c = output = input_emb
        #TODO:先用经过嵌入后的序列计算mamba需要的图

        # dist_graph = self.graph.adjacency_matrix().to_dense().to(device)
        # dist_graph = dist_graph.expand(num_steps, *dist_graph.shape)
        # sr = self.gcn_layer(dist_graph, input_emb)
        # sr(batch_size, in_steps, num_nodes, model_dim)
        sl = self.gcn_layer(graph, input_emb)
        # sr(batch_size, in_steps, num_nodes, model_dim)
        trans_graph = sl.transpose(2, 3) @ graph @ sl
        # trans_graph = None
        output = output.transpose(1, 2)
        output = output.reshape(batch_size * num_nodes, num_steps, -1)
        c = c.transpose(1, 2)  # (batch_size, num_nodes, T, emb_scan_dim)
        c = c.reshape(batch_size * num_nodes, num_steps, -1)
        for i, z in enumerate(x_loc):
            z = z.transpose(1, 2)  # (batch_size, num_nodes, T, emb_scan_dim)
            z = z.reshape(batch_size * num_nodes, num_steps, -1)
            # (batch_size * num_nodes, num_steps, emb_scan_dim)
            mamba_out = self.mamba_blocks[i](z, trans_graph)
            output += mamba_out * self.pws[i](c) * self.scale[i]
            c = mamba_out
        out_emb = output.reshape(batch_size, num_nodes, num_steps, -1)
        out_emb = out_emb.transpose(1, 2)
        # (batch_size, steps, num_nodes, emb_scan_dim)
        # batch_size, num_patch, num_nodes, steps

        # Spatial Mixing
        out_emb = out_emb.reshape(batch_size * num_steps, num_nodes, -1)
        out_emb = self.transformer_encoder(out_emb)
        out_emb = out_emb.view(batch_size, num_steps, num_nodes, -1)
        h = self.activation(out_emb)
        h = self.ff_scan(h)
        h = self.activation(h)

        # Regression Layer
        out = self.regression_layer(h)
        return out, graph, sl

