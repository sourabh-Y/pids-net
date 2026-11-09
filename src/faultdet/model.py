"""CNN-Transformer of Thomas et al. (IEEE TIM 2023), as implemented in the authors' code.

(B, 12800, C) -> 50 slices of 256 samples -> shared 1-D CNN -> (B, 50, 64)
-> + learned position embedding -> 2 transformer encoder blocks -> flatten (3200)
-> two dense heads (fault type, fault location).
"""
import math
import torch
import torch.nn as nn
from .backbones import BACKBONES, keras_init

class KerasMHA(nn.Module):
    """Keras MultiHeadAttention(num_heads, key_dim): heads x key_dim projections,
    output projected back to the model width (4 x 128 -> 64 here)."""
    def __init__(self, d, heads=4, key_dim=128, dropout=0.2):
        super().__init__()
        self.h, self.dk = heads, key_dim
        self.q, self.k, self.v = (nn.Linear(d, heads * key_dim) for _ in range(3))
        self.o = nn.Linear(heads * key_dim, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = (f(x).view(B, T, self.h, self.dk).transpose(1, 2) for f in (self.q, self.k, self.v))
        att = self.drop(torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), dim=-1))
        return self.o((att @ v).transpose(1, 2).reshape(B, T, self.h * self.dk))

    def reset_keras(self):
        """Keras Glorot init over the full einsum kernels: (d, heads, key_dim) for q/k/v and
        (heads, key_dim, d) for the output; much smaller q/k/v weights than nn.Linear's default."""
        d = self.q.in_features
        bq = math.sqrt(6.0 / (self.h * d + self.dk * d))
        for lin in (self.q, self.k, self.v):
            nn.init.uniform_(lin.weight, -bq, bq)
            nn.init.zeros_(lin.bias)
        bo = math.sqrt(6.0 / (self.dk * self.h + d * self.h))
        nn.init.uniform_(self.o.weight, -bo, bo)
        nn.init.zeros_(self.o.bias)

class EncoderBlock(nn.Module):
    """Authors' AttentionEncoder: e1 = LN(x); e2 = MHA(e1) + e1; out = FFN(LN(e2)) + e2."""
    def __init__(self, d=64, heads=4, key_dim=128, ff=128, dropout=0.2):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.mha = KerasMHA(d, heads, key_dim, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(ff, d), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        e1 = self.ln1(x)
        e2 = self.mha(e1) + e1
        return self.ff(self.ln2(e2)) + e2


def head(d_in, n):
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.ReLU(),
                         nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, n))


class CNNTransformer(nn.Module):
    def __init__(self, backbone='Xception', in_ch=15, n_type=15, n_loc=15, n_slices=50, d=64):
        super().__init__()
        self.n_slices = n_slices
        self.cnn = BACKBONES[backbone](in_ch)
        assert self.cnn.out_dim == d
        self.pos = nn.Embedding(n_slices, d)
        self.encoder = nn.Sequential(EncoderBlock(d), EncoderBlock(d))
        self.type_head, self.loc_head = head(n_slices * d, n_type), head(n_slices * d, n_loc)
        for m in (self.encoder, self.type_head, self.loc_head):
            keras_init(m)
        for block in self.encoder:
            block.mha.reset_keras()
        nn.init.uniform_(self.pos.weight, -0.05, 0.05)            # Keras Embedding default

    def forward(self, x):                                        # x: (B, 12800, C)
        B, L, C = x.shape
        n = self.n_slices
        s = x.view(B, n, L // n, C).permute(0, 1, 3, 2).reshape(B * n, C, L // n)
        f = self.cnn(s).view(B, n, -1) + self.pos.weight
        f = self.encoder(f).flatten(1)
        return self.type_head(f), self.loc_head(f)               # logits


def keras_param_count(model):
    """Parameter count as Keras reports it (trainable + BatchNorm moving statistics)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats = sum(b.numel() for n, b in model.named_buffers() if n.endswith(('running_mean', 'running_var')))
    return trainable + stats


if __name__ == '__main__':                                       # self-test: python -m faultdet.model
    x = torch.randn(2, 12800, 15)
    for name in BACKBONES:
        m = CNNTransformer(name).eval()
        with torch.no_grad():
            t, l = m(x)
        print(f'{name:18s} {keras_param_count(m):>12,d} params   outputs {tuple(t.shape)} {tuple(l.shape)}')