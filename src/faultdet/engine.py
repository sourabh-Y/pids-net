"""Training loop: mixed precision, checkpoint on validation loss, optional early stopping."""
import math
import time

import torch
import torch.nn as nn


def run_epoch(model, loader, device, opt=None, scaler=None, amp=True, clip=0, sched=None):
    train = opt is not None
    model.train(train)
    ce = nn.CrossEntropyLoss(reduction='sum')
    loss_sum, t_ok, l_ok, n = 0.0, 0, 0, 0
    for x, yt, yl in loader:
        x, yt, yl = x.to(device, non_blocking=True), yt.to(device), yl.to(device)
        with torch.set_grad_enabled(train), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            pt, pl = model(x)
        loss = ce(pt.float(), yt) + ce(pl.float(), yl)          # Keras: mean CE(type) + mean CE(loc)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss / len(x)).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
        loss_sum += loss.item()
        t_ok += (pt.argmax(1) == yt).sum().item()
        l_ok += (pl.argmax(1) == yl).sum().item()
        n += len(x)
    return {'loss': loss_sum / n, 'type_acc': t_ok / n, 'loc_acc': l_ok / n}


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    pt_all, pl_all = [], []
    for x, _, _ in loader:
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            pt, pl = model(x.to(device, non_blocking=True))
        pt_all.append(torch.softmax(pt.float(), 1).cpu())
        pl_all.append(torch.softmax(pl.float(), 1).cpu())
    return torch.cat(pt_all).double().numpy(), torch.cat(pl_all).double().numpy()


def fit(model, train_loader, val_loader, device, epochs, lr, ckpt, patience=0, amp=True, log=print, clip=0, warmup=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)        # Keras Adam defaults
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    sched = None
    if warmup:
        steps = warmup * len(train_loader)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps))
    best, best_ep, history = math.inf, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loader.dataset.set_epoch(ep)                              # fresh noise (if enabled)
        tr = run_epoch(model, train_loader, device, opt, scaler, amp, clip, sched)
        va = run_epoch(model, val_loader, device, amp=amp)
        history.append({'epoch': ep, **{f'train_{k}': v for k, v in tr.items()},
                        **{f'val_{k}': v for k, v in va.items()}})
        improved = va['loss'] < best
        if improved:
            best, best_ep = va['loss'], ep
            torch.save(model.state_dict(), ckpt)
        log(f"ep {ep:3d}  train {tr['loss']:.4f} ({tr['type_acc']:.3f}/{tr['loc_acc']:.3f})  "
            f"val {va['loss']:.4f} ({va['type_acc']:.3f}/{va['loc_acc']:.3f})  {time.time() - t0:.0f}s"
            + ('  *' if improved else ''))
        if patience and ep - best_ep >= patience:
            log(f'early stopping: best epoch {best_ep}')
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return history, best_ep