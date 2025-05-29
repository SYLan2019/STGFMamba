import  torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.layers import ComplexLinear
class GraphPropagate(nn.Module):
    def __init__(self, Ks, dropout = 0.2):
        super(GraphPropagate, self).__init__()
        self.Ks = Ks
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, graph):
        if self.Ks < 1:
            raise ValueError(
                f"ERROR: Ks must be a positive integer, received {self.Ks}."
            )
        x_k = x; x_list = [x]
        for k in range(1, self.Ks):
            # graph(step, num_nodes, num_nodes) x_k(batch_size, step, num_nodes, emb_dim)
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))

        return x_list
class GFMambaEncoder(nn.Module):
    def __init__(self, e_layers: int = 4,
                 d_model: int = 128,
                 gddmlp: bool = True,
                 d_ff: int = 128,
                 c_out: int = 12,
                 dt_rank: int = 32,
                 reduction: int = 2,
                 bias: bool = True,
                 dt_init: str = 'random',
                 d_state: int = 16,
                 dropout: float = 0.1,
                 dt_max: float = 0.1,
                 dt_min: float = 0.001,
                 dt_init_floor: float = 1e-4,
                 dt_scale: float = 1.0):
        super().__init__()

        self.layers = nn.ModuleList([GFMambaBlock(d_model = d_model,
                                                 d_ff = d_ff,
                                                 bias = bias,
                                                 dt_rank = dt_rank,
                                                 d_state = d_state,
                                                 dt_init = dt_init,
                                                 c_out = c_out,
                                                 reduction = reduction,
                                                 dropout = dropout,
                                                 dt_max = dt_max,
                                                 dt_min = dt_min,
                                                 dt_init_floor = dt_init_floor,
                                                 dt_scale = dt_scale) for _ in range(e_layers)])

    def forward(self, x, graph):
        # x : [bs * nvars, steps, d_model]

        for layer in self.layers:
            x = layer(x, graph)

        x = F.silu(x)

        return x

class GFMambaBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 d_ff: int = 128,
                 c_out: int = 12,
                 num_nodes: int =307,
                 dt_rank: int = 32,
                 reduction: int = 2,
                 bias: bool = True,
                 dt_init: str = 'random',
                 d_state: int = 16,
                 dropout: float = 0.1,
                 dt_max: float = 0.1,
                 dt_min: float = 0.001,
                 dt_init_floor: float = 1e-4,
                 dt_scale: float = 1.0
                 ):
        super().__init__()

        self.mixer = MambaBlock(d_model = d_model,
                                d_ff = d_ff,
                                bias = bias,
                                dt_rank = dt_rank,
                                d_state = d_state,
                                dt_init = dt_init,
                                dt_max = dt_max,
                                dt_min = dt_min,
                                dt_init_floor = dt_init_floor,
                                dt_scale = dt_scale)
        self.norm = RMSNorm(d_model)
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.c_out = c_out
        self.num_nodes = num_nodes
    def forward(self, x, graph):
        # x : [bs * nvars, steps, d_model]

        # output : [bs * nvars, steps, d_model]

        output = self.mixer(self.norm(x), graph)
        output = self.weight * output + self.bias
        output = self.dropout(output)
        output += x
        return output
class MambaBlock(nn.Module):
    """
    MambaModule, similar to https://arxiv.org/pdf/2402.18959
    """
    def __init__(self,
                 d_model: int = 128,
                 d_ff: int = 128,
                 bias: bool = True,
                 dt_rank: int = 32,
                 d_state: int = 16,
                 dt_init: str = 'random',
                 dt_max: float = 0.1,
                 dt_min: float = 0.001,
                 dt_init_floor: float = 1e-4,
                 dt_scale: float = 1.0,
                 padding_factor: float = 1.0,
                 u: float = 1e-6,
                 k_top: int = 6,
                 in_steps: int = 12):
        super().__init__()
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.d_ff = d_ff
        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(d_model, 2 * d_ff, bias=bias)

        # projects x to input-dependent Δ, B, C, D
        self.x_proj = nn.Linear(d_ff, dt_rank + 2 * d_state + d_ff, bias=False)

        # projects Δ from dt_rank to d_ff
        self.dt_proj = nn.Linear(dt_rank, d_ff, bias=True)
        self.p = padding_factor
        self.u = u
        self.k_top = k_top
        self.in_steps = in_steps
        stored_steps = in_steps // 2 + 1
        self.freq_linear = ComplexLinear(int(in_steps * (1+padding_factor) // 2 + 1 + k_top), stored_steps)
        # dt initialization
        # dt weights
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(d_ff) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)

        self.A_log = nn.Parameter(torch.log(A))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x, graph):
        # x : [bs * nvars, steps, d_model]

        # y : [bs * nvars, steps, d_model]

        bn, L, d = x.shape
        device = x.device

        x_padded = torch.cat((x, torch.zeros([bn, int(L * self.p), d]).to(device)), dim=1)
        fp = torch.fft.rfft(x_padded, dim=1)
        # fp(bs * nvars, steps*(1+p), d_model)
        f = torch.fft.rfft(x, dim=1)
        real_part = f.real
        imag_part = f.imag

        real_part_adjusted = real_part + self.u
        imag_part_adjusted = imag_part + self.u

        squared_sum_adjusted = real_part_adjusted**2 + imag_part_adjusted**2
        sorted_squared_sum, _ = torch.sort(squared_sum_adjusted, descending=True, dim=1)
        fs = sorted_squared_sum[:,:self.k_top,:]
        freq_proj = self.freq_linear(torch.cat((fp, fs), dim=1).transpose(1, 2)).transpose(1, 2)
        real_part = freq_proj.real
        imag_part = freq_proj.imag

        squared_sum = real_part**2 + imag_part**2
        wf = torch.softmax(squared_sum, dim=2)
        # f = f * self.plain_filter.view(1, -1, 1)
        x_freq = torch.fft.irfft(wf * f, dim=1)

        xz = self.in_proj(x) # [bs * nvars, steps, 2 * d_ff]
        x, z = xz.chunk(2, dim=-1) # [bs * nvars, steps, d_ff], [bs * nvars, steps, d_ff]

        # x branch
        x = F.silu(x)
        y = self.ssm(x, graph)

        # z branch
        z = F.silu(z)

        output = y * z * x_freq
        output = self.out_proj(output) # [bs * nvars, steps, d_ff]

        return output

    def ssm(self, x, graph):
        # x : [bs * nvars, steps, d_ff]

        # y : [bs * nvars, steps, d_ff]

        A = -torch.exp(self.A_log.float()) # [d_ff, d_state]

        deltaBCD = self.x_proj(x) # [bs * nvars, steps, dt_rank + 2 * d_state + d_ff]
        # [bs * nvars, steps, dt_rank], [bs * nvars, steps, d_state], [bs * nvars, steps, d_state], [bs * nvars, steps, d_ff]
        delta, B, C, D = torch.split(deltaBCD, [self.dt_rank, self.d_state, self.d_state, self.d_ff], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) # [bs * nvars, steps, d_ff]
        batch_size, _, _, _ = graph.shape # (batch_size, in_steps, model_dim, model_dim)
        _, steps, dim = delta.shape
        delta = delta.reshape(batch_size, -1, steps, dim)
        delta = torch.einsum('bnsd, bsda->bnsa', delta, graph)
        delta = delta.reshape(-1, steps, dim)
        #(batch_size * num_nodes, steps, dim)
        y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : [bs * nvars, steps, d_ff]
        # Δ : [bs * nvars, steps, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, steps, d_state]
        # C : [bs * nvars, steps, d_state]
        # D : [bs * nvars, steps, d_ff]

        # y : [bs * nvars, steps, d_ff]

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [bs * nvars, steps, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, steps, d_ff, d_state]

        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, steps, d_ff, d_state]

        h = torch.zeros(x.size(0), self.d_ff, self.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1) # [bs * nvars, steps, d_ff, d_state]
        # [bs * nvars, steps, d_ff, d_state] @ [bs * nvars, steps, d_state, 1] -> [bs * nvars, steps, d_ff, 1]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output




