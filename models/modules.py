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
            # hi与ij维度进行矩阵乘法
            # graph(step, num_nodes, num_nodes) x_k(batch_size, step, num_nodes, emb_dim)
            x_k = torch.einsum("thi,btij->bthj", graph, x_k.clone())
            x_list.append(self.dropout(x_k))

        return x_list
class CMambaEncoder(nn.Module):
    def __init__(self, e_layers: int = 4,
                 d_model: int = 128,
                 gddmlp: bool = True,
                 d_ff: int = 128,
                 c_out: int = 12,
                 dt_rank: int = 32,
                 reduction: int = 2,
                 avg: bool = False,
                 bias: bool = True,
                 max: bool = False,
                 dt_init: str = 'random',
                 d_state: int = 16,
                 dropout: float = 0.1,
                 dt_max: float = 0.1,
                 dt_min: float = 0.001,
                 dt_init_floor: float = 1e-4,
                 dt_scale: float = 1.0):
        super().__init__()

        self.layers = nn.ModuleList([CMambaBlock(d_model = d_model,
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
                                                 dt_scale = dt_scale) for _ in range(e_layers)])

    def forward(self, x, graph):
        # x : [bs * nvars, patch_num, d_model]

        for layer in self.layers:
            x = layer(x, graph)

        x = F.silu(x)

        return x

class CMambaBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 gddmlp: bool = True,
                 d_ff: int = 128,
                 c_out: int = 12,
                 num_nodes: int =307,
                 dt_rank: int = 32,
                 reduction: int = 2,
                 avg: bool = False,
                 bias: bool = True,
                 max: bool = False,
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
        self.weight = nn.Parameter(torch.tensor(0.5))  # 标量weight
        self.bias = nn.Parameter(torch.tensor(0.5))   # 标量bias
        self.gddmlp = gddmlp
        if self.gddmlp:
            print("Insert Norm")
            # self.GDDMLP = GDDMLP(c_out, reduction,
            #                      avg, max)

        self.dropout = nn.Dropout(dropout)
        self.c_out = c_out
        self.num_nodes = num_nodes
    def forward(self, x, graph):
        # x : [bs * nvars, steps, d_model]

        # output : [bs * nvars, steps, d_model]

        output = self.mixer(self.norm(x), graph)

        if self.gddmlp:
            # # output : [bs, nvars, patch_num, d_model]
            # output = self.GDDMLP(output.reshape(-1, self.num_nodes,
            #                                     output.shape[-2], output.shape[-1]))
            # # output : [bs * nvars, patch_num, d_model]
            # output = output.reshape(-1, output.shape[-2], output.shape[-1])
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
        print('振幅有频域，减topk')
        print(f'top k:{k_top}')
        # self.fp_linear = ComplexLinear(int(in_steps * (1+padding_factor) // 2 + 1), stored_steps)
        # self.fs_linear = ComplexLinear(k_top, stored_steps)
        # self.plain_filter = nn.Parameter(torch.Tensor(stored_steps))

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
        # x : [bs * nvars, patch_num, d_model]

        # y : [bs * nvars, patch_num, d_model]

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
#        #==============振幅拼接===========
        sorted_squared_sum, _ = torch.sort(squared_sum_adjusted, descending=True, dim=1)
        fs = sorted_squared_sum[:,:self.k_top,:]
#        #===============================
#        # fs(bs * nvars, self.k_top, d_model)
#        #==============频率拼接===========
#        _, indices = torch.topk(squared_sum_adjusted, dim=1, k=self.k_top)
#        fs = torch.gather(f, dim=1, index=indices)
#        #===============================
#        #=============双振幅拼接=========
#        # fp = (fp.real+self.u)**2 + (fp.imag+self.u)**2
#        # freq_proj = self.freq_amp_linear(torch.cat((fp, fs), dim=1).transpose(1, 2)).transpose(1, 2)
#        #==============================
#        # fs(bs * nvars, self.k_top, d_model)
#        # freq_proj = self.fp_linear(fp.transpose(1, 2)).transpose(1, 2) + self.fs_linear(fs.transpose(1, 2)).transpose(1, 2)
        freq_proj = self.freq_linear(torch.cat((fp, fs), dim=1).transpose(1, 2)).transpose(1, 2)
        real_part = freq_proj.real
        imag_part = freq_proj.imag

        squared_sum = real_part**2 + imag_part**2
        wf = torch.softmax(squared_sum, dim=2)
        # f = f * self.plain_filter.view(1, -1, 1)
        x_freq = torch.fft.irfft(wf * f, dim=1)

        xz = self.in_proj(x) # [bs * nvars, patch_num, 2 * d_ff]
        x, z = xz.chunk(2, dim=-1) # [bs * nvars, patch_num, d_ff], [bs * nvars, patch_num, d_ff]

        # x branch
        x = F.silu(x)
        y = self.ssm(x, graph)

        # z branch
        z = F.silu(z)

        output = y * z * x_freq
#        output = y * z
        output = self.out_proj(output) # [bs * nvars, patch_num, d_ff]

        return output

    def ssm(self, x, graph):
        # x : [bs * nvars, patch_num, d_ff]

        # y : [bs * nvars, patch_num, d_ff]

        A = -torch.exp(self.A_log.float()) # [d_ff, d_state]

        deltaBCD = self.x_proj(x) # [bs * nvars, patch_num, dt_rank + 2 * d_state + d_ff]
        # [bs * nvars, patch_num, dt_rank], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_ff]
        delta, B, C, D = torch.split(deltaBCD, [self.dt_rank, self.d_state, self.d_state, self.d_ff], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) # [bs * nvars, patch_num, d_ff]
        if graph is not None:
            batch_size, _, _, _ = graph.shape # (batch_size, in_steps, model_dim, model_dim)
            _, steps, dim = delta.shape
            delta = delta.reshape(batch_size, -1, steps, dim)
            # delta(batch_size, num_nodes, steps, dim)
            # graph(batch_size, steps, dim, dim)  'bnsd, bda->bnsa'
            delta = torch.einsum('bnsd, bsda->bnsa', delta, graph)

            # delta(batch_size, num_nodes, steps, dim)
            # graph(batch_size, steps, num_nodes, num_nodes)
            # delta = torch.einsum('bsnm, bmsd->bnsd', graph, delta)
            # graph左乘delta应该不能替代delta左乘graph,并且相乘的维度应该要在dim上
            delta = delta.reshape(-1, steps, dim)
        #(batch_size * num_nodes, steps, dim)
        y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]

        # y : [bs * nvars, patch_num, d_ff]

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]

        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]

        h = torch.zeros(x.size(0), self.d_ff, self.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1) # [bs * nvars, patch_num, d_ff, d_state]
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff, 1]
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
        # x(9824, 12, 108)
        return output
class GDDMLP(nn.Module):
    def __init__(self, n_vars, reduction=2, avg_flag=True, max_flag=True):
        super().__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        print(f'avg_flag:{avg_flag}, max_flag:{max_flag}')
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc_sc = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.fc_sf = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.sigmoid = nn.Sigmoid()

        #self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fc_sc:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)

        for layer in self.fc_sf:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)

    def forward(self, x):
        b, n, p, d = x.shape
        scale = torch.zeros_like(x)
        shift = torch.zeros_like(x)
        if self.avg_flag:
            sc = self.fc_sc(self.avg_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            sf = self.fc_sf(self.avg_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        if self.max_flag:
            sc = self.fc_sc(self.max_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            sf = self.fc_sf(self.max_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        return self.sigmoid(scale) * x + self.sigmoid(shift)

class GatedGDDMLP(nn.Module):
    def __init__(self, n_vars, reduction=2, d_model = 128, avg_flag=True, max_flag=True, bias = True):
        super().__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        print(f'avg_flag:{avg_flag}, max_flag:{max_flag}')
        self.fc_sc = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.fc_sf = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        #self.initialize_weights()
    def forward(self, x):
        b, n, p, d = x.shape
        z = self.in_proj(x)
        z =F.gelu(z)
        z1, z2 = z.chunk(2, dim=-1)
        scale = torch.zeros_like(z2)
        shift = torch.zeros_like(z2)
        if self.avg_flag:
            pooled = self.avg_pool(z2.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)
            sc = self.fc_sc(pooled).permute(0, 2, 1)
            sf = self.fc_sf(pooled).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        if self.max_flag:
            pooled = self.max_pool(z2.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)
            sc = self.fc_sc(pooled).permute(0, 2, 1)
            sf = self.fc_sf(pooled).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)

        out = self.sigmoid(scale) * z1 + self.sigmoid(shift)
        out = self.out_proj(out)
        return out

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, batch_norm=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, groups=g, dilation=d, bias=False)
        self.do_bn = False
        if batch_norm:
            self.bn = nn.BatchNorm1d(c2)
            self.do_bn = True
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        if self.do_bn:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.conv(x))


    def forward_fuse(self, x):
        return self.act(self.conv(x))
class SCAM(nn.Module):
    def __init__(self, in_channels, n_vars, in_steps = 12, reduction=2):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_steps, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv(in_steps, in_steps, 1, 1, batch_norm=False)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # GMP

        self.fc_avg = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.fc_max = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                                   nn.GELU(),
                                   nn.Linear(n_vars // reduction, n_vars, bias=False))

    def forward(self, x):
        # n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        b, n, p, d = x.shape
        x = x.reshape(b * n, p, d)  # (B*N, C, L)
        # avg max: [b, n, p, 1]
        avg = self.avg_pool(x).reshape(b, n, p).permute(0, 2, 1)
        max = self.max_pool(x).reshape(b, n, p).permute(0, 2, 1)
        # avg max: [b, n, p]
        avg = self.fc_avg(avg).permute(0, 2, 1).sigmoid()
        max = self.fc_max(max).permute(0, 2, 1).sigmoid()
        # k: [b * n, d, 1]
        k = self.k(x).permute(0, 2, 1).softmax(1)
        # v: [b * n, p, d]
        v = self.v(x.permute(0, 2, 1)).permute(0, 2, 1)
        # y: [b * n, p, 1]
        y = torch.matmul(v, k)
        # y_pool:[b*n, 1, d]
        y_pool = torch.matmul((max + avg).reshape(b*n, 1, p), v)
        return (y_pool * y).reshape(b, n, p, d)
