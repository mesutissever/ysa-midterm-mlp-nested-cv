"""Bağımsız test değerlendirmesi (ödev Bölüm 13) — yalnızca kilitli commit doğrulandıktan sonra.

Kurallar:
  - Kilitli tahmin dosyası (results/test_predictions_locked.csv) olduğu gibi okunur; model yeniden
    eğitilmez, eşik (0.50) ve ön işleme değiştirilmez.
  - Test etiketleri yalnızca metrik hesaplamak için okunur; hiçbir seçim/ayar yapılmaz.
  - Ana model: nihai MLP, ana seed (final_seeds[0]). Diğer seed'ler yalnızca kararlılık raporu içindir.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.data import TARGET, get_dev_test
from src.final import LOCKED_PATH, RESULTS_DIR, THRESHOLD, check_lock_committed
from src.nested_cv import METRIC_COLS, compute_metrics

BOOT_METRICS = {
    "AP": lambda y, p: average_precision_score(y, p),
    "ROC_AUC": lambda y, p: roc_auc_score(y, p),
    "F1": lambda y, p: f1_score(y, (p >= THRESHOLD).astype(int), zero_division=0),
}


# ----------------------------------------------------------------------------- veri
def load_locked_with_labels():
    """Kilit doğrulanır -> kilitli tahminler + test etiketleri + test özellikleri (orig_index ile hizalı)."""
    lock = check_lock_committed()                     # commit yoksa / dosya değiştiyse hata verir
    locked = pd.read_csv(LOCKED_PATH)
    df_dev, df_test = get_dev_test()
    assert len(locked) == len(df_test) and locked["orig_index"].is_unique
    assert set(locked["orig_index"]) == set(df_test.index), "Kilitli dosya ile test indeksleri uyuşmuyor!"
    df = locked.merge(df_test, left_on="orig_index", right_index=True, how="left", validate="one_to_one")
    assert df[TARGET].notna().all()
    # Dummy (prior): development pozitif oranı — öğrenilen tek şey bu, test'e bakmaz
    df["prob_dummy"] = float(df_dev[TARGET].mean())
    df["pred_dummy"] = (df["prob_dummy"] >= THRESHOLD).astype(int)
    return df, lock


def model_columns(seeds):
    p = seeds["final_seeds"][0]
    return {f"MLP (M2, seed {p})": f"prob_mlp_s{p}", "LogisticRegression": "prob_logreg",
            "Dummy (prior)": "prob_dummy"}


# ----------------------------------------------------------------------------- metrikler
def test_metrics(df, seeds, save=True):
    rows = [dict(model=name, **compute_metrics(df[TARGET], df[col], THRESHOLD))
            for name, col in model_columns(seeds).items()]
    out = pd.DataFrame(rows)
    if save:
        out.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    return out


def bootstrap_ci(df, seeds, n_boot=1000, alpha=0.05, save=True):
    """Yüzdelik bootstrap %95 GA (AP, ROC-AUC, F1) + eşleştirilmiş fark (MLP - LogReg).
    Tek sınıf içeren yeniden örneklemler atlanır ve sayılır."""
    y = df[TARGET].to_numpy()
    cols = model_columns(seeds)
    probs = {m: df[c].to_numpy() for m, c in cols.items()}
    mlp, lr = list(cols)[0], "LogisticRegression"
    rng = np.random.RandomState(seeds["bootstrap"])
    n = len(y)
    stats = {(m, k): [] for m in probs for k in BOOT_METRICS}
    diffs = {k: [] for k in BOOT_METRICS}
    skipped = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        vals = {(m, k): f(yb, probs[m][idx]) for m in probs for k, f in BOOT_METRICS.items()}
        for key, v in vals.items():
            stats[key].append(v)
        for k in BOOT_METRICS:
            diffs[k].append(vals[(mlp, k)] - vals[(lr, k)])

    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    rows = []
    for (m, k), v in stats.items():
        lo, hi = np.percentile(v, q)
        rows.append(dict(model=m, metric=k, point=BOOT_METRICS[k](y, probs[m]), ci_low=lo, ci_high=hi,
                         boot_sd=np.std(v, ddof=1), n_valid=len(v), n_skipped=skipped))
    ci = pd.DataFrame(rows)
    drows = []
    for k, v in diffs.items():
        lo, hi = np.percentile(v, q)
        point = BOOT_METRICS[k](y, probs[mlp]) - BOOT_METRICS[k](y, probs[lr])
        drows.append(dict(comparison=f"{mlp} - {lr}", metric=k, diff=point, ci_low=lo, ci_high=hi,
                          p_boot_le0=float(np.mean(np.asarray(v) <= 0)),
                          ci_contains_0=bool(lo <= 0 <= hi), n_valid=len(v)))
    diff = pd.DataFrame(drows)
    if save:
        ci.to_csv(RESULTS_DIR / "test_bootstrap_ci.csv", index=False)
        diff.to_csv(RESULTS_DIR / "test_bootstrap_diff.csv", index=False)
    return ci, diff


def seed_stability(df, seeds, save=True):
    rows = [dict(seed=s, primary=(s == seeds["final_seeds"][0]),
                 **compute_metrics(df[TARGET], df[f"prob_mlp_s{s}"], THRESHOLD))
            for s in seeds["final_seeds"]]
    per_seed = pd.DataFrame(rows)
    mean, sd = per_seed[METRIC_COLS].mean(), per_seed[METRIC_COLS].std(ddof=1)
    summary = pd.DataFrame({"metric": METRIC_COLS, "mean": mean.values, "sd": sd.values,
                            "min": per_seed[METRIC_COLS].min().values,
                            "max": per_seed[METRIC_COLS].max().values})
    summary["mean_sd"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(summary["mean"], summary["sd"])]
    if save:
        per_seed.to_csv(RESULTS_DIR / "seed_stability_results.csv", index=False)
        summary.to_csv(RESULTS_DIR / "seed_stability_summary.csv", index=False)
    return per_seed, summary


# ----------------------------------------------------------------------------- hata analizi
ERROR_FEATURES = ["age", "job", "contact", "month", "campaign", "pdays", "previous", "poutcome",
                  "emp.var.rate", "euribor3m", "nr.employed"]


def error_analysis(df, prob_col, top_k=10, save=True):
    """FP/FN listesi (en emin hatalar), kategori bazında hata oranları ve betimsel kalibrasyon."""
    pred = (df[prob_col] >= THRESHOLD).astype(int)
    kind = np.select([(pred == 1) & (df[TARGET] == 0), (pred == 0) & (df[TARGET] == 1),
                      (pred == 1) & (df[TARGET] == 1)], ["FP", "FN", "TP"], "TN")
    d = df.assign(prob=df[prob_col], pred=pred, outcome=kind)
    keep = ["orig_index", TARGET, "prob", "pred", "outcome"] + ERROR_FEATURES
    top_fp = d[d.outcome == "FP"].nlargest(top_k, "prob")[keep]
    top_fn = d[d.outcome == "FN"].nsmallest(top_k, "prob")[keep]
    errors = pd.concat([top_fp, top_fn], ignore_index=True)

    # Pozitif sınıfın kaçırılması hangi gruplarda yoğun? (gerçek pozitifler içinde FN oranı)
    pos = d[d[TARGET] == 1]
    by_cat = []
    for c in ["poutcome", "contact", "month"]:
        g = pos.groupby(c, observed=True)["outcome"].agg(n_pos="size", n_FN=lambda s: (s == "FN").sum())
        g["FN_rate"] = g["n_FN"] / g["n_pos"]
        by_cat.append(g.reset_index().rename(columns={c: "level"}).assign(feature=c))
    by_cat = pd.concat(by_cat, ignore_index=True)[["feature", "level", "n_pos", "n_FN", "FN_rate"]]

    # Betimsel kalibrasyon: olasılık beşte birlik dilimlerinde ortalama tahmin vs gözlenen oran
    bins = pd.qcut(d["prob"], 5, duplicates="drop")
    calib = d.groupby(bins, observed=True).agg(n=("prob", "size"), mean_prob=("prob", "mean"),
                                               observed_rate=(TARGET, "mean")).reset_index()
    calib["prob"] = calib["prob"].astype(str)
    counts = d["outcome"].value_counts().reindex(["TP", "FP", "TN", "FN"]).fillna(0).astype(int)
    if save:
        errors.to_csv(RESULTS_DIR / "test_error_analysis.csv", index=False)
        by_cat.to_csv(RESULTS_DIR / "test_fn_by_category.csv", index=False)
        calib.to_csv(RESULTS_DIR / "test_calibration_bins.csv", index=False)
    return dict(counts=counts, errors=errors, fn_by_category=by_cat, calibration=calib,
                mean_prob=float(d["prob"].mean()), pos_rate=float(d[TARGET].mean()))
