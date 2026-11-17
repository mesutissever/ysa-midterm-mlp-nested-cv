"""Deney konfigürasyonları: eğitim ayarları, M1–M6, arama uzayı ve R1–R6.

Bu dosya `python -m src.configs` ile BİR KEZ çalıştırılır ve configs/ altındaki JSON'lar
nested CV başlamadan önce commit edilir. Nested CV yalnızca JSON'ları okur; yeniden örnekleme yapmaz.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

from src.data import load_seeds
from src.train import ES_FRACTION, ES_PATIENCE, MAX_EPOCHS, SCHED, SGD_MOMENTUM

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs"

TRAINING = dict(
    max_epochs=MAX_EPOCHS,
    early_stopping=dict(patience=ES_PATIENCE, restore_best_weights=True, min_delta=0.0,
                        monitor="internal validation BCE loss (mode=min)",
                        monitor_fraction=ES_FRACTION, monitor_split="stratified"),
    lr_scheduler=dict(name="ReduceLROnPlateau", **SCHED),
    sgd_momentum=SGD_MOMENTUM,
    loss="BCEWithLogitsLoss", output="single logit + sigmoid",
    outer_cv=dict(name="StratifiedKFold", n_splits=3, shuffle=True),
    inner_cv=dict(name="StratifiedKFold", n_splits=3, shuffle=True),
    selection_metric="average_precision_score (AP), inner CV mean",
    threshold=0.5, class_weight=None, device="cpu",
)

# Ortak varsayılanlar (ödev Bölüm 9)
MANUAL_BASE = dict(optimizer="adamw", hidden=[32], activation="relu", lr=1e-3,
                   weight_decay=1e-4, batch_size=64, scaler="standard", dropout=0.0,
                   clip=False, scheduler=True)

MANUAL = {
    "M1": dict(changes={}, purpose="Temel manuel konfigürasyon"),
    "M2": dict(changes={"optimizer": "adam"}, purpose="Optimizer etkisi (AdamW -> Adam)"),
    "M3": dict(changes={"optimizer": "sgd"}, purpose="Optimizer etkisi (AdamW -> SGD+Momentum 0.9)"),
    "M4": dict(changes={"scaler": "minmax"}, purpose="Ölçekleme etkisi (Standard -> MinMax)"),
    "M5": dict(changes={"hidden": [64, 32]}, purpose="Mimari etkisi ((32,) -> (64,32))"),
    "M6": dict(changes={"scheduler": False}, purpose="LR düşürme etkisi (scheduler kapalı)"),
}

# Random Search arama uzayı (ödev Bölüm 10)
SEARCH_SPACE = dict(
    optimizer=["sgd", "adam", "adamw"],
    candidates_per_optimizer=2,
    hidden=[[16], [32], [64], [64, 32], [128, 64]],
    activation=["relu", "tanh"],
    lr={"adam": [1e-4, 1e-2], "adamw": [1e-4, 1e-2], "sgd": [1e-3, 1e-1], "scale": "log-uniform"},
    weight_decay=[0.0, 1e-5, 1e-4, 1e-3],
    batch_size=[32, 64, 128],
    scaler=["standard", "minmax", "robust"],
    dropout=[0.0, 0.2],
    clip=[False, True],
    fixed=dict(scheduler=True),
    sampler="sklearn.model_selection.ParameterSampler, tek numpy RandomState akışı (seed=random_search)",
)


def build_manual():
    return {k: dict(code=k, **{**MANUAL_BASE, **v["changes"]}, purpose=v["purpose"])
            for k, v in MANUAL.items()}


def sample_random(seed):
    """Her optimizer için 2 aday; tek RandomState akışı sırayla sgd -> adam -> adamw."""
    rng = np.random.RandomState(seed)
    out, i = {}, 1
    for opt in SEARCH_SPACE["optimizer"]:
        lo, hi = SEARCH_SPACE["lr"][opt]
        dist = {
            "hidden": SEARCH_SPACE["hidden"], "activation": SEARCH_SPACE["activation"],
            "lr": loguniform(lo, hi), "weight_decay": SEARCH_SPACE["weight_decay"],
            "batch_size": SEARCH_SPACE["batch_size"], "scaler": SEARCH_SPACE["scaler"],
            "dropout": SEARCH_SPACE["dropout"], "clip": SEARCH_SPACE["clip"],
        }
        for p in ParameterSampler(dist, n_iter=SEARCH_SPACE["candidates_per_optimizer"], random_state=rng):
            code = f"R{i}"
            out[code] = dict(code=code, optimizer=opt, hidden=list(p["hidden"]),
                             activation=p["activation"], lr=float(f"{p['lr']:.4g}"),
                             weight_decay=float(p["weight_decay"]), batch_size=int(p["batch_size"]),
                             scaler=p["scaler"], dropout=float(p["dropout"]), clip=bool(p["clip"]),
                             **SEARCH_SPACE["fixed"])
            i += 1
    return out


def validate_random(cands):
    opts = [c["optimizer"] for c in cands.values()]
    assert len(cands) == 6
    for o in SEARCH_SPACE["optimizer"]:
        assert opts.count(o) == 2, f"{o}: {opts.count(o)} aday"
    for c in cands.values():
        lo, hi = SEARCH_SPACE["lr"][c["optimizer"]]
        assert lo <= c["lr"] <= hi, c
        assert c["hidden"] in SEARCH_SPACE["hidden"]
        for k in ["activation", "weight_decay", "batch_size", "scaler", "dropout", "clip"]:
            assert c[k] in SEARCH_SPACE[k], (k, c[k])
    return True


def load_json(name):
    with open(CFG_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_candidates():
    """Nested CV için: {'manual': {...}, 'random': {...}} — yalnızca kayıtlı JSON'lardan okur."""
    return {"manual": load_json("manual_configs.json"), "random": load_json("random_configs.json")}


def _dump(obj, name):
    with open(CFG_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    seeds = load_seeds()
    rnd_path = CFG_DIR / "random_configs.json"
    if rnd_path.exists():
        raise SystemExit("random_configs.json zaten var; adaylar sabittir, yeniden üretilmez.")
    manual = build_manual()
    random_c = sample_random(seeds["random_search"])
    validate_random(random_c)
    _dump(TRAINING, "training.json")
    _dump(manual, "manual_configs.json")
    _dump(SEARCH_SPACE, "search_space.json")
    _dump(random_c, "random_configs.json")
    for c in list(manual.values()) + list(random_c.values()):
        print({k: v for k, v in c.items() if k != "purpose"})
    print("OK: her optimizer için 2 aday, tüm değerler arama uzayı içinde.")
