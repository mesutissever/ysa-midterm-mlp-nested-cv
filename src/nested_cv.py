"""3×3 nested CV: manuel (M1–M6) ve otomatik (R1–R6) arama, referans modeller, development seçimi.

Akış (ödev Bölüm 5, 11, 12.1):
  Outer StratifiedKFold(3) -> her outer-train içinde inner StratifiedKFold(3)
    - 12 aday aynı iç katlarda eğitilir (train_mlp kendi içinde %85 fit / %15 early-stop ayırır)
    - Manuel kazanan = M1–M6 içinde en yüksek iç CV mean AP; otomatik kazanan = R1–R6 içinde
    - Kazananlar outer-train'de yeniden eğitilir, outer-eval'de BİR KEZ ölçülür
    - Dummy ve LogisticRegression outer-train'de eğitilir, outer-eval'de ölçülür
  Son olarak tüm development üzerinde inner CV (3 kat × 12 aday) -> nihai aday (en yüksek mean AP)

Eşitlik kuralı (önceden tanımlı): mean AP büyük -> parametre sayısı küçük -> kod sırası.
Outer-eval skorları hiçbir seçimde kullanılmaz.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import TARGET
from src.preprocessing import build_logreg_preprocessor
from src.train import MAX_EPOCHS, predict_proba, train_mlp

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
HP_KEYS = ["optimizer", "hidden", "activation", "lr", "weight_decay", "batch_size",
           "scaler", "dropout", "clip", "scheduler"]
LOGREG_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=2000)


# ----------------------------------------------------------------------------- metrikler
def compute_metrics(y_true, prob, threshold=0.5):
    """AP (birincil), ROC-AUC ve eşik (varsayılan 0.50) tabanlı metrikler."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(
        AP=average_precision_score(y_true, prob),
        ROC_AUC=roc_auc_score(y_true, prob),
        Accuracy=accuracy_score(y_true, pred),
        Precision=precision_score(y_true, pred, zero_division=0),
        Sensitivity=recall_score(y_true, pred, zero_division=0),
        Specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        F1=f1_score(y_true, pred, zero_division=0),
        Balanced_Accuracy=balanced_accuracy_score(y_true, pred),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
    )


METRIC_COLS = ["AP", "ROC_AUC", "Accuracy", "Precision", "Sensitivity", "Specificity",
               "F1", "Balanced_Accuracy"]


# ----------------------------------------------------------------------------- yardımcılar
def split_xy(df):
    return df.drop(columns=TARGET), df[TARGET]


def summarize_inner(rows):
    """İç CV satırlarını aday bazında özetler (mean AP ± örnek SD, süre, en iyi epoch...)."""
    df = pd.DataFrame(rows)
    g = df.groupby("code", sort=False)
    out = pd.DataFrame({
        "mean_AP": g["AP"].mean(), "sd_AP": g["AP"].std(ddof=1),
        "mean_ROC_AUC": g["ROC_AUC"].mean(),
        "mean_time_s": g["train_time_s"].mean(), "total_time_s": g["train_time_s"].sum(),
        "mean_best_epoch": g["best_epoch"].mean(), "mean_stopped_epoch": g["stopped_epoch"].mean(),
        "n_params": g["n_params"].mean(), "n_warnings": g["warnings"].apply(lambda s: (s != "").sum()),
        "n_failed": g["failed"].sum(),
    }).reset_index()
    out["order"] = range(len(out))
    return out


def select_best(summary, prefix=None):
    """Önceden tanımlı kural: mean AP büyük -> n_params küçük -> kod sırası."""
    s = summary if prefix is None else summary[summary["code"].str.startswith(prefix)]
    s = s.sort_values(["mean_AP", "n_params", "order"], ascending=[False, True, True])
    return s.iloc[0]["code"]


def run_inner_cv(cands, X, y, n_splits, cv_seed, seed, es_seed, max_epochs, tag, log):
    """Tüm adayları aynı iç katlarda eğitir; her (aday, kat) için iç validation AP'sini döndürür."""
    rows, hists = [], []
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=cv_seed).split(X, y))
    for code, cfg in cands.items():
        for k, (tr, va) in enumerate(folds, 1):
            res = train_mlp(cfg, X.iloc[tr], y.iloc[tr], seed=seed, es_seed=es_seed,
                            max_epochs=max_epochs)
            prob = predict_proba(res, X.iloc[va])
            yv = y.iloc[va].values
            rows.append(dict(**tag, inner_fold=k, code=code, n_inner_train=len(tr), n_inner_val=len(va),
                             AP=average_precision_score(yv, prob), ROC_AUC=roc_auc_score(yv, prob),
                             **res.summary))
            hists.append(res.history.assign(**tag, stage="inner", inner_fold=k, code=code))
        last = rows[-1]
        log(f"  {tag} {code}: mean AP={np.mean([r['AP'] for r in rows if r['code'] == code]):.4f} "
            f"(son kat best_ep={last['best_epoch']}, {last['warnings'] or 'uyarı yok'})")
    return rows, hists


def fit_baselines(X_tr, y_tr, X_ev, seed):
    """Dummy (prior) ve sabit ön işlemeli LogisticRegression; yalnızca X_tr'de fit."""
    out = {}
    t = time.perf_counter()
    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    out["dummy"] = (dummy.predict_proba(X_ev)[:, 1], time.perf_counter() - t)
    t = time.perf_counter()
    logreg = Pipeline([("pre", build_logreg_preprocessor()),
                       ("clf", LogisticRegression(**LOGREG_PARAMS, random_state=seed))]).fit(X_tr, y_tr)
    out["logreg"] = (logreg.predict_proba(X_ev)[:, 1], time.perf_counter() - t)
    return out, logreg


def mean_sd_table(outer_df):
    """Dış kat sonuçlarını yöntem bazında 'Mean ± SD (n=3)' olarak özetler."""
    g = outer_df.groupby("method", sort=False)[METRIC_COLS + ["train_time_s"]]
    mean, sd = g.mean(), g.std(ddof=1)
    fmt = mean.copy().astype(object)
    for c in mean.columns:
        fmt[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], sd[c])]
    return mean, sd, fmt


# ----------------------------------------------------------------------------- ana akış
def run_nested_cv(df_dev, cands, seeds, max_epochs=MAX_EPOCHS, save=True, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    all_cands = {**cands["manual"], **cands["random"]}
    X, y = split_xy(df_dev)
    seed, es_seed = seeds["torch_init"], seeds["early_stop_split"]
    t_start = time.perf_counter()

    inner_rows, hists, outer_rows, preds, selections = [], [], [], [], []
    outer = StratifiedKFold(3, shuffle=True, random_state=seeds["outer_cv"])

    for of, (otr, oev) in enumerate(outer.split(X, y), 1):
        log(f"\n=== DIŞ KAT {of}: outer-train={len(otr)}, outer-eval={len(oev)} ===")
        X_otr, y_otr, X_oev, y_oev = X.iloc[otr], y.iloc[otr], X.iloc[oev], y.iloc[oev]

        rows, h = run_inner_cv(all_cands, X_otr, y_otr, 3, seeds["inner_cv"], seed, es_seed,
                               max_epochs, {"outer_fold": of}, log)
        inner_rows += rows
        hists += h
        summ = summarize_inner(rows)
        winners = {"manual_search": select_best(summ, "M"), "random_search": select_best(summ, "R")}
        log(f"  Seçilen: manuel={winners['manual_search']}, otomatik={winners['random_search']}")

        for method, code in winners.items():
            res = train_mlp(all_cands[code], X_otr, y_otr, seed=seed, es_seed=es_seed, max_epochs=max_epochs)
            prob = predict_proba(res, X_oev)
            m = summ.set_index("code").loc[code]
            outer_rows.append(dict(outer_fold=of, method=method, selected=code,
                                   inner_mean_AP=m["mean_AP"], inner_sd_AP=m["sd_AP"],
                                   **compute_metrics(y_oev, prob),
                                   train_time_s=res.summary["train_time_s"],
                                   best_epoch=res.summary["best_epoch"],
                                   stopped_epoch=res.summary["stopped_epoch"],
                                   n_params=res.summary["n_params"], warnings=res.summary["warnings"]))
            hists.append(res.history.assign(outer_fold=of, stage="outer_refit", inner_fold=0, code=code))
            preds.append(pd.DataFrame({"outer_fold": of, "orig_index": X_oev.index, "y": y_oev.values,
                                       "method": method, "selected": code, "prob": prob}))

        base, _ = fit_baselines(X_otr, y_otr, X_oev, seed)
        for method, (prob, tsec) in base.items():
            outer_rows.append(dict(outer_fold=of, method=method, selected=method,
                                   **compute_metrics(y_oev, prob), train_time_s=tsec))
            preds.append(pd.DataFrame({"outer_fold": of, "orig_index": X_oev.index, "y": y_oev.values,
                                       "method": method, "selected": method, "prob": prob}))
        selections.append(dict(outer_fold=of, **winners))

    # --- Tüm development üzerinde inner CV -> nihai seçim (outer sonuçlarına bakılmaz)
    log("\n=== TÜM DEVELOPMENT ÜZERİNDE İÇ CV (nihai seçim) ===")
    rows, h = run_inner_cv(all_cands, X, y, 3, seeds["inner_cv"], seed, es_seed,
                           max_epochs, {"outer_fold": 0}, log)
    inner_rows += rows
    hists += h
    dev_summ = summarize_inner(rows)
    dev_sel = dict(manual_best=select_best(dev_summ, "M"), random_best=select_best(dev_summ, "R"),
                   final=select_best(dev_summ))
    fin = dev_summ.set_index("code").loc[dev_sel["final"]]
    dev_sel.update(final_procedure="manual_search" if dev_sel["final"].startswith("M") else "random_search",
                   final_dev_inner_mean_AP=float(fin["mean_AP"]), final_dev_inner_sd_AP=float(fin["sd_AP"]),
                   final_config=all_cands[dev_sel["final"]],
                   rule="Tüm development inner CV (3 kat) mean AP en yüksek; eşitlikte n_params küçük, sonra kod sırası")
    log(f"  Manuel en iyi={dev_sel['manual_best']}, otomatik en iyi={dev_sel['random_best']}, "
        f"NİHAİ={dev_sel['final']} (mean AP={fin['mean_AP']:.4f})")

    total = time.perf_counter() - t_start
    log(f"\nToplam süre: {total/60:.1f} dk, MLP eğitimi: {len(inner_rows) + 2*3} (iç {len(inner_rows)} + refit 6)")

    out = dict(inner=pd.DataFrame(inner_rows), outer=pd.DataFrame(outer_rows),
               preds=pd.concat(preds, ignore_index=True), hist=pd.concat(hists, ignore_index=True),
               selections=pd.DataFrame(selections), dev_selection=dev_sel, total_time_s=total)
    if save:
        save_results(out, all_cands)
    return out


def _search_table(inner, all_cands, prefix):
    """Aday × kapsam (outer 1–3 iç CV, dev iç CV) tablosu + hiperparametreler."""
    tabs = []
    for of, g in inner[inner["code"].str.startswith(prefix)].groupby("outer_fold"):
        s = summarize_inner(g.to_dict("records")).drop(columns="order")
        s.insert(0, "scope", "development" if of == 0 else f"outer_{of}")
        tabs.append(s)
    t = pd.concat(tabs, ignore_index=True)
    hp = pd.DataFrame([{**{k: all_cands[c][k] for k in HP_KEYS}, "code": c} for c in t["code"].unique()])
    hp["hidden"] = hp["hidden"].apply(lambda h: str(tuple(h)))
    return t.merge(hp, on="code")


def save_results(out, all_cands, results_dir=RESULTS_DIR):
    results_dir.mkdir(exist_ok=True)
    inner = out["inner"]
    inner.to_csv(results_dir / "inner_cv_results.csv", index=False)
    _search_table(inner, all_cands, "M").to_csv(results_dir / "manual_search_results.csv", index=False)
    _search_table(inner, all_cands, "R").to_csv(results_dir / "random_search_results.csv", index=False)
    out["outer"].to_csv(results_dir / "outer_cv_results.csv", index=False)
    mean, sd, fmt = mean_sd_table(out["outer"])
    summary = pd.concat({"mean": mean, "sd": sd}, axis=1)
    summary.to_csv(results_dir / "outer_cv_summary.csv")
    fmt.to_csv(results_dir / "outer_cv_summary_formatted.csv")
    out["preds"].to_csv(results_dir / "outer_predictions.csv", index=False)
    out["hist"].to_csv(results_dir / "training_histories.csv", index=False)
    out["selections"].to_csv(results_dir / "outer_selections.csv", index=False)
    with open(results_dir / "dev_selection.json", "w", encoding="utf-8") as f:
        json.dump({**out["dev_selection"], "total_runtime_s": out["total_time_s"]}, f,
                  ensure_ascii=False, indent=2, default=float)
