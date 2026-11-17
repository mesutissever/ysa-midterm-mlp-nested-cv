"""Tek bir MLP eğitimi: fit / early-stop ayrımı, kat-içi ön işleme, early stopping, ReduceLROnPlateau.

train_mlp() içindeki veri kullanımı (ödev Bölüm 5, 8):
  X_train (ör. inner-train)
    ├─ %85 FIT       -> ön işleme fit + MLP ağırlık güncellemesi
    └─ %15 EARLY-STOP -> yalnızca transform; BCE kaybı early stopping ve scheduler'ı yönetir
  Inner validation / outer evaluation / test bu fonksiyona HİÇ verilmez;
  çağıran kod sonradan predict_proba() ile skorlar.
"""
import copy
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from src.model import MLP, count_params
from src.preprocessing import build_preprocessor, fit_transform_split

MAX_EPOCHS = 150
ES_PATIENCE = 15
ES_FRACTION = 0.15
SCHED = dict(mode="min", factor=0.5, patience=5, min_lr=1e-5)
SGD_MOMENTUM = 0.9

DEFAULT_CONFIG = dict(
    optimizer="adamw", hidden=(32,), activation="relu", lr=1e-3, weight_decay=1e-4,
    batch_size=64, scaler="standard", dropout=0.0, clip=False, scheduler=True,
)


def set_seed(seed):
    """Python, NumPy ve PyTorch seed'lerini ayarlar; deterministik algoritmaları zorlar."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_optimizer(name, params, lr, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=SGD_MOMENTUM, weight_decay=weight_decay)
    if name == "adam":        # weight decay = L2 cezası gradyana eklenir (coupled)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":       # weight decay ağırlığa doğrudan uygulanır (decoupled)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Bilinmeyen optimizer: {name}")


@dataclass
class TrainResult:
    model: nn.Module
    preprocessor: object
    history: pd.DataFrame
    summary: dict = field(default_factory=dict)


def train_mlp(config, X_train, y_train, seed, es_seed, max_epochs=MAX_EPOCHS, verbose=False):
    """config ile bir MLP eğitir. X_train: DataFrame (ham özellikler), y_train: 0/1 Series."""
    cfg = {**DEFAULT_CONFIG, **config}
    t0 = time.perf_counter()

    # 1) Fit / early-stop ayrımı (stratified, sabit seed)
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=ES_FRACTION, stratify=y_train, random_state=es_seed)

    # 2) Ön işleme yalnızca FIT alt kümesinde öğrenilir
    pre = build_preprocessor(cfg["scaler"], cfg["clip"])
    Z_fit, Z_es = fit_transform_split(pre, X_fit, X_es)
    Xf, yf = torch.from_numpy(Z_fit), torch.tensor(y_fit.values, dtype=torch.float32)
    Xe, ye = torch.from_numpy(Z_es), torch.tensor(y_es.values, dtype=torch.float32)

    # 3) Seed, model, optimizer, scheduler
    set_seed(seed)
    model = MLP(Z_fit.shape[1], tuple(cfg["hidden"]), cfg["activation"], cfg["dropout"])
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **SCHED) if cfg["scheduler"] else None
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_epoch, best_state, wait = math.inf, 0, None, 0
    history, warnings = [], []
    n, bs = len(Xf), int(cfg["batch_size"])

    # 4) Epoch döngüsü
    for epoch in range(1, max_epochs + 1):
        te = time.perf_counter()
        lr_now = opt.param_groups[0]["lr"]
        model.train()
        perm = torch.randperm(n, generator=gen)
        run_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
        train_loss = run_loss / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xe), ye).item()

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            lr=lr_now, epoch_time_s=time.perf_counter() - te))

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            warnings.append(f"NaN/inf kayıp (epoch {epoch})")
            break

        if val_loss < best_loss:                      # min_delta = 0
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        if sched is not None:
            sched.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"ep {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr_now:.2e}")

        if wait >= ES_PATIENCE:
            break

    stopped_epoch = history[-1]["epoch"]
    early_stopped = wait >= ES_PATIENCE
    if not early_stopped and stopped_epoch == max_epochs:
        warnings.append("Max epoch'a ulaştı (early stopping tetiklenmedi)")

    failed = best_state is None
    if not failed:
        model.load_state_dict(best_state)             # best weights restore
    else:
        warnings.append("Eğitim başarısız: geçerli bir epoch yok")

    summary = dict(
        n_params=count_params(model), n_inputs=Z_fit.shape[1],
        n_fit=len(X_fit), n_es=len(X_es),
        best_epoch=best_epoch, stopped_epoch=stopped_epoch, early_stopped=early_stopped,
        best_val_loss=best_loss, final_lr=opt.param_groups[0]["lr"],
        train_time_s=time.perf_counter() - t0, failed=failed, warnings="; ".join(warnings),
    )
    return TrainResult(model, pre, pd.DataFrame(history), summary)


def predict_proba(result, X):
    """Fit edilmiş ön işleme (yalnızca transform) + model -> pozitif sınıf olasılığı."""
    Z = torch.from_numpy(result.preprocessor.transform(X).astype(np.float32))
    result.model.eval()
    with torch.no_grad():
        return torch.sigmoid(result.model(Z)).numpy()
