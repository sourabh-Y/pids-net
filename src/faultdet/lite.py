"""Lightweight fault classifier for relay-rate current windows.

(B, T, 15 * groups) globally scaled I / dI windows, e.g. T = 128 at 7.7 kHz
-> Frontend: fixed, parameter-free
     Clarke transform per monitoring point (a, b, c -> alpha, beta, 0; 0 = (a + b + c) / 3)
     per-sample scaling of each feature group (removes fault size, which Rf controls)
-> small residual depthwise-separable 1-D CNN -> mean + max pooling -> two heads (type, location)

TwoStream gives each task its own input: fault type from I and dI, location from dI only.
The Frontend can also be put in front of the paper's CNN-Transformer (WithFrontend), so that
input processing and architecture can be compared separately.
"""
import math

import torch
import torch.nn as nn

CLARKE = [[2 / 3, -1 / 3, -1 / 3],                   # alpha
          [0.0, 1 / math.sqrt(3), -1 / math.sqrt(3)],   # beta
          [1 / 3, 1 / 3, 1 / 3]]                    # zero sequence


class Frontend(nn.Module):
    """x: (B, T, 15 * groups), channels [a b c] x 5 monitoring points per group.
    Returns the transformed x and log of each group's per-sample scale (B, groups)."""

    def __init__(self, n_groups, clarke=True, norm='sample', eps=1e-6):
        super().__init__()
        self.g, self.clarke, self.norm, self.eps = n_groups, clarke, norm, eps
        self.register_buffer('C', torch.tensor(CLARKE).T.contiguous(), persistent=False)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):   # scaling in fp32
            B, T, _ = x.shape
            x = x.float().reshape(B, T, self.g, 5, 3)
            if self.clarke:
                x = x @ self.C
            s = x.abs().amax(dim=(1, 3, 4))                 # (B, groups)
            if self.norm == 'sample':
                x = x / (s[:, None, :, None, None] + self.eps)
            return x.reshape(B, T, -1), torch.log(s + self.eps)


class SepBlock(nn.Module):
    """Residual block: (depthwise k, pointwise) x 2 with BatchNorm; 1x1 strided shortcut."""

    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        p = k // 2
        self.f = nn.Sequential(
            nn.Conv1d(cin, cin, k, stride, p, groups=cin, bias=False), nn.BatchNorm1d(cin),
            nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.SiLU(),
            nn.Conv1d(cout, cout, k, 1, p, groups=cout, bias=False), nn.BatchNorm1d(cout),
            nn.Conv1d(cout, cout, 1, bias=False), nn.BatchNorm1d(cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.f(x) + self.skip(x))


def _head(d_in, n, hidden=128, dropout=0.2):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, n))


class LiteNet(nn.Module):
    WIDTHS = (32, 32, 48, 64, 96, 128)                   # stem, then 5 blocks (strides 1, 2, 2, 2, 2)

    def __init__(self, in_ch, n_groups, clarke=True, norm='sample', logscale=False, width=1.0,
                 n_type=15, n_loc=15, heads=('type', 'loc')):
        super().__init__()
        self.front, self.logscale = Frontend(n_groups, clarke, norm), logscale
        w = [max(8, int(round(c * width))) for c in self.WIDTHS]
        self.stem = nn.Sequential(nn.Conv1d(in_ch, w[0], 7, 1, 3, bias=False), nn.BatchNorm1d(w[0]), nn.SiLU())
        self.blocks = nn.Sequential(*[SepBlock(w[i], w[i + 1], 1 if i == 0 else 2) for i in range(5)])
        d = 2 * w[-1] + (n_groups if logscale else 0)
        self.type_head = _head(d, n_type) if 'type' in heads else None
        self.loc_head = _head(d, n_loc) if 'loc' in heads else None

    def forward(self, x):                                # x: (B, T, C)
        x, logs = self.front(x)
        h = self.blocks(self.stem(x.transpose(1, 2)))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        if self.logscale:
            h = torch.cat([h, logs.to(h.dtype)], 1)
        return (self.type_head(h) if self.type_head is not None else None,
                self.loc_head(h) if self.loc_head is not None else None)   # logits


class TwoStream(nn.Module):
    """Task-specific inputs: the type head sees `type_groups` (default I and dI), the location
    head sees `loc_groups` (default dI only). Each stream is a separate LiteNet trunk with its own
    frontend; x carries all `groups` (15 channels each, in that order)."""

    def __init__(self, groups, type_groups, loc_groups, clarke=True, norm='sample', ls_type=False,
                 ls_loc=False, width=1.0, n_type=15, n_loc=15):
        super().__init__()
        ch = lambda gs: torch.tensor([15 * groups.index(g) + c for g in gs for c in range(15)])
        self.register_buffer('ti', ch(type_groups), persistent=False)
        self.register_buffer('li', ch(loc_groups), persistent=False)
        self.t = LiteNet(15 * len(type_groups), len(type_groups), clarke, norm, ls_type, width,
                         n_type, n_loc, heads=('type',))
        self.l = LiteNet(15 * len(loc_groups), len(loc_groups), clarke, norm, ls_loc, width,
                         n_type, n_loc, heads=('loc',))

    def forward(self, x):
        return self.t(x[..., self.ti])[0], self.l(x[..., self.li])[1]


class WithFrontend(nn.Module):
    """Paper's CNN-Transformer behind the same Frontend (cross-check of input processing)."""

    def __init__(self, front, net):
        super().__init__()
        self.front, self.net = front, net

    def forward(self, x):
        return self.net(self.front(x)[0])


@torch.no_grad()
def count_macs(model, x):
    """Multiply-accumulates per sample of all Conv1d and Linear layers (attention products not included)."""
    macs, hooks, B = [0], [], x.shape[0]       # per input sample (the paper's CNN sees B x 50 slices)

    def conv(m, i, o):
        macs[0] += o.numel() // B * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin(m, i, o):
        macs[0] += o.numel() // B * m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin))
    was = model.training
    model.eval()
    model(x)
    model.train(was)
    for h in hooks:
        h.remove()
    return macs[0]


if __name__ == '__main__':                               # self-test: python -m faultdet.lite
    from .model import keras_param_count
    for g, T, kw in [(2, 128, {}), (2, 84, {}), (1, 128, dict(clarke=False, norm='global')),
                     (2, 128, dict(logscale=True))]:
        m = LiteNet(15 * g, g, **kw).eval()
        x = torch.randn(4, T, 15 * g)
        t, l = m(x)
        print(f'groups {g} T {T:3d} {kw}: {keras_param_count(m):,d} params, '
              f'{count_macs(m, x[:1]) / 1e6:.2f} M MACs, outputs {tuple(t.shape)} {tuple(l.shape)}')
    m = TwoStream(['I', 'dI'], ['I', 'dI'], ['dI']).eval()
    x = torch.randn(4, 128, 30)
    t, l = m(x)
    print(f'two-stream (type: I+dI, location: dI): {keras_param_count(m):,d} params, '
          f'{count_macs(m, x[:1]) / 1e6:.2f} M MACs, outputs {tuple(t.shape)} {tuple(l.shape)}')
