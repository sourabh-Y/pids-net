"""Model size and speed: parameters, MACs and batch-1 latency per decision window.

    python scripts/latency.py --gpu 0 [--threads 1] [--out results/latency.csv]
CPU latency is measured with --threads torch threads (1 = a single core, closest to an embedded relay).
GPU latency uses the same mixed precision as training. Median over repeated runs after warm-up.
Lite models include their input processing (Clarke + scaling). 'PIDS-Net, from 769 kHz' also includes
the block averaging and the one-cycle difference, starting from the raw 12 800-sample windows.
The CPU model, PyTorch version and thread count are written to <out>_env.txt.
"""
import argparse
import platform
import csv
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faultdet.lite import LiteNet, TwoStream, count_macs  # noqa: E402
from faultdet.model import CNNTransformer, keras_param_count  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=0, help='-1: CPU only')
p.add_argument('--threads', type=int, default=1)
p.add_argument('--min_time', type=float, default=3.0, help='seconds of timed runs per model and device')
p.add_argument('--out', default=os.path.join('results', 'latency.csv'))
o = p.parse_args()
torch.set_num_threads(o.threads)

class FromRaw(torch.nn.Module):
    """PIDS-Net fed with the 769 kHz windows: block mean of `factor` samples and one-cycle difference included."""

    def __init__(self, net, factor=100):
        super().__init__()
        self.net, self.factor = net, factor

    def forward(self, x):                            # x: (B, 12800, 30) = I_now and I_prev at the simulation rate
        b, t, c = x.shape
        d = x.view(b, t // self.factor, self.factor, c).mean(2)
        now, prev = d[..., :15], d[..., 15:]
        return self.net(torch.cat([now, now - prev], -1))


def cpu_name():
    try:
        if platform.system() == 'Windows':
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        if os.path.exists('/proc/cpuinfo'):
            for line in open('/proc/cpuinfo'):
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or 'unknown'


MODELS = [  # name, constructor, input shape (window, channels)
    ('Xception (paper)', lambda: CNNTransformer('Xception', 15), (12800, 15)),
    ('DenseNet-169 (paper)', lambda: CNNTransformer('DenseNet169', 15), (12800, 15)),
    ('Inception-ResNet-v2 (paper)', lambda: CNNTransformer('InceptionResNetV2', 15), (12800, 15)),
    ('EfficientNet-B2 (paper)', lambda: CNNTransformer('EfficientNetB2', 15), (12800, 15)),
    ('Lite (I+dI, 1 stream)', lambda: LiteNet(30, 2), (128, 30)),
    ('Lite2 (two-stream)', lambda: TwoStream(['I', 'dI'], ['I', 'dI'], ['dI']), (128, 30)),
    ('Lite2, half cycle', lambda: TwoStream(['I', 'dI'], ['I', 'dI'], ['dI']), (84, 30)),
    ('Lite2, from 769 kHz', lambda: FromRaw(TwoStream(['I', 'dI'], ['I', 'dI'], ['dI'])), (12800, 30)),
]


@torch.no_grad()
def timeit(model, x, amp):
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    run = lambda: model(x)
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
        for _ in range(3):                                   # warm-up
            run()
        sync()
        times, t_end = [], time.perf_counter() + o.min_time
        while time.perf_counter() < t_end or len(times) < 5:
            t0 = time.perf_counter()
            run()
            sync()
            times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


rows = []
print(f'{"model":30s} {"params":>10s} {"MACs":>10s} {f"CPU x{o.threads} ms":>12s} {"GPU ms":>8s}')
for name, make, shape in MODELS:
    m = make().eval()
    x = torch.randn(1, *shape)
    par, macs = keras_param_count(m), count_macs(m, x)
    cpu = timeit(m, x, amp=False)
    gpu = float('nan')
    if o.gpu >= 0 and torch.cuda.is_available():
        dev = torch.device(f'cuda:{o.gpu}')
        torch.backends.cudnn.benchmark = True
        gpu = timeit(m.to(dev), x.to(dev), amp=True)
    rows.append([name, par, macs, shape[0], cpu, gpu])
    print(f'{name:30s} {par / 1e6:9.3f}M {macs / 1e9:9.4f}G {cpu:12.2f} {gpu:8.2f}', flush=True)

os.makedirs(os.path.dirname(o.out) or '.', exist_ok=True)
with open(o.out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['model', 'params', 'macs', 'window_samples', f'cpu_ms_{o.threads}thread', 'gpu_ms'])
    w.writerows(rows)
env = [f'CPU: {cpu_name()}', f'threads: {o.threads}', f'PyTorch: {torch.__version__}',
       f'Python: {platform.python_version()}', f'OS: {platform.platform()}']
if o.gpu >= 0 and torch.cuda.is_available():
    env.append(f'GPU: {torch.cuda.get_device_name(o.gpu)}, CUDA {torch.version.cuda}')
with open(os.path.splitext(o.out)[0] + '_env.txt', 'w') as fh:
    fh.write('\n'.join(env) + '\n')
print('\n'.join(env))
print('saved', o.out)
