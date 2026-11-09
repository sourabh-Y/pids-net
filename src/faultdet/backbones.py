"""1-D CNN backbones, ported layer by layer from the authors' Keras code (1d_deep_cnns.py).

Each backbone maps (N, C, 256) -> (N, 64): the authors replaced the ImageNet tops by
1x1 conv blocks ending in 64 channels, followed by global average pooling.
Keras conventions are kept: 'same' padding (asymmetric when strided), BatchNorm
momentum 0.99 / per-layer epsilon, frozen gamma where Keras uses scale=False,
Glorot-uniform weights (EfficientNet: VarianceScaling(2, fan_out, normal)).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
def same_pad(x, k, s, value=0.0):
    """TensorFlow 'same' padding for a 1-D window of size k and stride s."""
    length = x.shape[-1]
    total = max((math.ceil(length / s) - 1) * s + k - length, 0)
    return F.pad(x, (total // 2, total - total // 2), value=value)


class BN(nn.BatchNorm1d):
    """Keras BatchNormalization (momentum 0.99); scale=False freezes gamma at 1."""
    def __init__(self, c, eps=1e-3, scale=True):
        super().__init__(c, eps=eps, momentum=0.01)
        if not scale:
            self.weight.requires_grad_(False)


class Conv(nn.Module):
    """Conv1d with Keras padding semantics ('valid' or 'same')."""
    def __init__(self, cin, cout, k, s=1, padding='valid', bias=False, groups=1):
        super().__init__()
        self.k, self.s, self.same = k, s, padding == 'same'
        self.conv = nn.Conv1d(cin, cout, k, s, bias=bias, groups=groups)

    def forward(self, x):
        return self.conv(same_pad(x, self.k, self.s) if self.same else x)


class SepConv(nn.Module):
    """Keras SeparableConv1D ('same', no bias): depthwise then pointwise."""
    def __init__(self, cin, cout, k):
        super().__init__()
        self.dw = Conv(cin, cin, k, padding='same', groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class MaxPoolSame(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self.k, self.s = k, s

    def forward(self, x):
        return F.max_pool1d(same_pad(x, self.k, self.s, value=float('-inf')), self.k, self.s)


def keras_init(module):
    """Keras defaults: Glorot-uniform kernels, zero biases."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------- Xception
class Xception1D(nn.Module):
    def __init__(self, cin):
        super().__init__()
        r = nn.ReLU
        self.stem = nn.Sequential(Conv(cin, 32, 3, 2), BN(32), r(), Conv(32, 64, 3), BN(64), r())
        self.entry, c = nn.ModuleList(), 64
        for i, cout in enumerate((128, 256, 728)):
            main = nn.Sequential(*([] if i == 0 else [r()]),
                                 SepConv(c, cout, 3), BN(cout), r(), SepConv(cout, cout, 3), BN(cout),
                                 MaxPoolSame(3, 2))
            res = nn.Sequential(Conv(c, cout, 1, 2, 'same'), BN(cout))
            self.entry.append(nn.ModuleList([main, res]))
            c = cout
        self.middle = nn.ModuleList(
            nn.Sequential(*[m for _ in range(3) for m in (r(), SepConv(728, 728, 3), BN(728))])
            for _ in range(8))
        self.exit_main = nn.Sequential(r(), SepConv(728, 728, 3), BN(728), r(), SepConv(728, 1024, 3),
                                       BN(1024), MaxPoolSame(3, 2))
        self.exit_res = nn.Sequential(Conv(728, 1024, 1, 2, 'same'), BN(1024))
        self.top = nn.Sequential(SepConv(1024, 256, 3), BN(256), r(), SepConv(256, 64, 3), BN(64), r())
        self.out_dim = 64
        keras_init(self)

    def forward(self, x):
        x = self.stem(x)
        for main, res in self.entry:
            x = main(x) + res(x)
        for block in self.middle:
            x = x + block(x)
        x = self.exit_main(x) + self.exit_res(x)
        return self.top(x).mean(-1)


# ----------------------------------------------------------------------------- DenseNet-169
class DenseLayer(nn.Module):
    def __init__(self, cin, growth=32):
        super().__init__()
        self.f = nn.Sequential(BN(cin, 1.001e-5), nn.ReLU(), nn.Conv1d(cin, 4 * growth, 1, bias=False),
                               BN(4 * growth, 1.001e-5), nn.ReLU(), Conv(4 * growth, growth, 3, padding='same'))

    def forward(self, x):
        return torch.cat([x, self.f(x)], 1)


class DenseNet1D(nn.Module):
    def __init__(self, cin, blocks=(6, 12, 32, 32)):            # DenseNet-169
        super().__init__()
        layers, c = [nn.ConstantPad1d(3, 0.0), nn.Conv1d(cin, 64, 7, 2, bias=False), BN(64, 1.001e-5),
                     nn.ReLU(), nn.ConstantPad1d(1, 0.0), nn.MaxPool1d(3, 2)], 64
        for i, n in enumerate(blocks):
            for _ in range(n):
                layers.append(DenseLayer(c))
                c += 32
            if i < len(blocks) - 1:                              # transition: halve channels and length
                layers += [BN(c, 1.001e-5), nn.ReLU(), nn.Conv1d(c, c // 2, 1, bias=False), nn.AvgPool1d(2, 2)]
                c //= 2
        for cout in (640, 256, 64):                              # authors' final 1x1 conv blocks
            layers += [nn.Conv1d(c, cout, 1, bias=False), BN(cout, scale=False), nn.ReLU()]
            c = cout
        self.body, self.out_dim = nn.Sequential(*layers), 64
        keras_init(self)

    def forward(self, x):
        return self.body(x).mean(-1)


# ----------------------------------------------------------------------------- Inception-ResNet-v2
def cbn(cin, cout, k, s=1, padding='same'):
    """Authors' inc_res_conv1d_bn: Conv (no bias) -> BN (no scale) -> ReLU."""
    return nn.Sequential(Conv(cin, cout, k, s, padding), BN(cout, scale=False), nn.ReLU())


class IRBlock(nn.Module):
    def __init__(self, c, kind, scale, act=True):
        super().__init__()
        if kind == 'block35':
            br, cm = [cbn(c, 32, 1), nn.Sequential(cbn(c, 32, 1), cbn(32, 32, 3)),
                      nn.Sequential(cbn(c, 32, 1), cbn(32, 48, 3), cbn(48, 64, 3))], 128
        elif kind == 'block17':
            br, cm = [cbn(c, 192, 1), nn.Sequential(cbn(c, 128, 1), cbn(128, 160, 7), cbn(160, 192, 1))], 384
        else:                                                    # block8
            br, cm = [cbn(c, 192, 1), nn.Sequential(cbn(c, 192, 1), cbn(192, 224, 3), cbn(224, 256, 1))], 448
        self.branches, self.up = nn.ModuleList(br), nn.Conv1d(cm, c, 1, bias=True)
        self.scale, self.act = scale, act

    def forward(self, x):
        x = x + self.scale * self.up(torch.cat([b(x) for b in self.branches], 1))
        return F.relu(x) if self.act else x


class InceptionResNetV2_1D(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(cbn(cin, 32, 3, 2, 'valid'), cbn(32, 32, 3, padding='valid'), cbn(32, 64, 3),
                                  nn.MaxPool1d(3, 2), cbn(64, 80, 1, padding='valid'),
                                  cbn(80, 192, 3, padding='valid'), nn.MaxPool1d(3, 2))
        self.mixed_5b = nn.ModuleList([
            cbn(192, 96, 1),
            nn.Sequential(cbn(192, 48, 1), cbn(48, 64, 5)),
            nn.Sequential(cbn(192, 64, 1), cbn(64, 96, 3), cbn(96, 96, 3)),
            nn.Sequential(nn.AvgPool1d(3, 1, 1, count_include_pad=False), cbn(192, 64, 1))])
        self.block35 = nn.Sequential(*[IRBlock(320, 'block35', 0.17) for _ in range(10)])
        self.mixed_6a = nn.ModuleList([
            cbn(320, 384, 3, 2, 'valid'),
            nn.Sequential(cbn(320, 256, 1), cbn(256, 256, 3), cbn(256, 384, 3, 2, 'valid')),
            nn.MaxPool1d(3, 2)])
        self.block17 = nn.Sequential(*[IRBlock(1088, 'block17', 0.1) for _ in range(20)])
        self.mixed_7a = nn.ModuleList([
            nn.Sequential(cbn(1088, 256, 1), cbn(256, 384, 3, 2, 'valid')),
            nn.Sequential(cbn(1088, 256, 1), cbn(256, 288, 3, 2, 'valid')),
            nn.Sequential(cbn(1088, 256, 1), cbn(256, 288, 3), cbn(288, 320, 3, 2, 'valid')),
            nn.MaxPool1d(3, 2)])
        self.block8 = nn.Sequential(*[IRBlock(2080, 'block8', 0.2) for _ in range(9)],
                                    IRBlock(2080, 'block8', 1.0, act=False))
        self.top = nn.Sequential(cbn(2080, 1024, 1), cbn(1024, 256, 1), cbn(256, 64, 1))
        self.out_dim = 64
        keras_init(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.block35(torch.cat([b(x) for b in self.mixed_5b], 1))
        x = self.block17(torch.cat([b(x) for b in self.mixed_6a], 1))
        x = self.block8(torch.cat([b(x) for b in self.mixed_7a], 1))
        return self.top(x).mean(-1)


# ----------------------------------------------------------------------------- EfficientNet-B2
EFF_BLOCKS = [(3, 1, 32, 16, 1, 1), (3, 2, 16, 24, 6, 2), (5, 2, 24, 40, 6, 2), (3, 3, 40, 80, 6, 2),
              (5, 3, 80, 112, 6, 1), (5, 4, 112, 192, 6, 2), (3, 1, 192, 320, 6, 1)]
# (kernel, repeats, filters_in, filters_out, expand_ratio, stride); squeeze-excite ratio 0.25


def round_filters(f, width, divisor=8):
    f *= width
    new = max(divisor, int(f + divisor / 2) // divisor * divisor)
    return int(new + divisor) if new < 0.9 * f else int(new)


def correct_pad(length, k):                                      # authors' eff_correct_pad
    return (k // 2 - (1 - length % 2), k // 2)


class DropPath(nn.Module):
    """Keras Dropout(noise_shape=(None, 1, 1)): drops the whole residual branch per sample."""
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        return x * x.new_empty(x.shape[0], 1, 1).bernoulli_(keep) / keep


class MBConv(nn.Module):
    def __init__(self, cin, cout, k, s, expand, se=0.25, drop=0.0):
        super().__init__()
        c = cin * expand
        self.expand = (nn.Sequential(nn.Conv1d(cin, c, 1, bias=False), BN(c), nn.SiLU())
                       if expand != 1 else nn.Identity())
        self.k, self.s = k, s
        self.dw = nn.Sequential(nn.Conv1d(c, c, k, s, groups=c, bias=False), BN(c), nn.SiLU())
        cse = max(1, int(cin * se))
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(c, cse, 1), nn.SiLU(),
                                nn.Conv1d(cse, c, 1), nn.Sigmoid())
        self.project = nn.Sequential(nn.Conv1d(c, cout, 1, bias=False), BN(cout))
        self.skip = s == 1 and cin == cout
        self.drop = DropPath(drop) if self.skip and drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.expand(x)
        h = F.pad(h, correct_pad(h.shape[-1], self.k)) if self.s == 2 else same_pad(h, self.k, 1)
        h = self.dw(h)
        h = self.project(h * self.se(h))
        return x + self.drop(h) if self.skip else h


class EfficientNetB2_1D(nn.Module):
    def __init__(self, cin, width=1.1, depth=1.2, drop_connect=0.2):
        super().__init__()
        c = round_filters(32, width)
        self.stem = nn.Sequential(nn.Conv1d(cin, c, 3, 2, bias=False), BN(c), nn.SiLU())
        total, blocks, b = float(sum(blk[1] for blk in EFF_BLOCKS)), [], 0   # as in the authors' code
        for k, rep, fin, fout, e, s in EFF_BLOCKS:
            fin, fout = round_filters(fin, width), round_filters(fout, width)
            for j in range(math.ceil(depth * rep)):
                blocks.append(MBConv(fin if j == 0 else fout, fout, k, s if j == 0 else 1, e, 0.25,
                                     drop_connect * b / total))
                b += 1
        self.blocks = nn.Sequential(*blocks)
        self.top = nn.Sequential(nn.Conv1d(fout, 144, 1, bias=False), BN(144), nn.SiLU(),
                                 nn.Conv1d(144, 64, 1, bias=False), BN(64), nn.SiLU())
        self.out_dim = 64
        for m in self.modules():                                 # VarianceScaling(2.0, fan_out, normal)
            if isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels // m.groups
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(F.pad(x, correct_pad(x.shape[-1], 3)))
        return self.top(self.blocks(x)).mean(-1)


BACKBONES = {'DenseNet169': DenseNet1D, 'Xception': Xception1D,
             'InceptionResNetV2': InceptionResNetV2_1D, 'EfficientNetB2': EfficientNetB2_1D}