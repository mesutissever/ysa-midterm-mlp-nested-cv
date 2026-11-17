"""Nihai eğitim (tüm development, 3 seed), model/ön işleme kaydı ve kilitli (etiketsiz) test tahminleri.

Ödev Bölüm 12:
  - Nihai konfigürasyon tüm development iç CV'si ile seçilmiştir (results/dev_selection.json).
  - Model, development içinden ayrılan %15 early-stop alt kümesiyle eğitilir; test etiketlerine erişilmez.
  - Üç önceden belirlenmiş seed (configs/seeds.json -> final_seeds; ilki ana seed).
  - Test olasılıkları ve sınıf tahminleri etiket sütunu OLMADAN test_predictions_locked.csv'ye yazılır,
    ayrı bir commit ile GitHub'a yüklenir; metrikler yalnızca bu commit'ten sonra hesaplanır.
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from src.data import TARGET
from src.nested_cv import fit_baselines, split_xy
from src.train import predict_proba, train_mlp

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOCKED_PATH = RESULTS_DIR / "test_predictions_locked.csv"
THRESHOLD = 0.5


# ----------------------------------------------------------------------------- git yardımcıları
def _git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def git_state():
    _, head = _git("rev-parse", "HEAD")
    _, dirty = _git("status", "--porcelain", "--", "src", "configs")
    return dict(code_commit=head, src_or_configs_uncommitted=bool(dirty))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- kayıt yardımcıları
def learned_stats(pre):
    """Fit edilmiş ColumnTransformer'dan okunabilir dönüşüm istatistikleri."""
    out = {}
    for name, pipe, cols in pre.transformers_:
        if name == "remainder":
            continue
        d = {"columns": list(cols)}
        for step_name, step in pipe.named_steps.items():
            if step_name == "impute":
                d["impute_" + step.strategy] = [v if isinstance(v, str) else float(v) for v in step.statistics_]
            elif step_name == "clip":
                d["clip_lower"], d["clip_upper"] = step.lower_.tolist(), step.upper_.tolist()
            elif step_name == "scale":
                for attr in ["mean_", "scale_", "center_", "data_min_", "data_max_"]:
                    if hasattr(step, attr) and getattr(step, attr) is not None:
                        d[f"scaler_{attr.rstrip('_')}"] = np.asarray(getattr(step, attr)).tolist()
                d["scaler"] = type(step).__name__
            elif step_name == "onehot":
                d["categories"] = {c: list(map(str, cats)) for c, cats in zip(cols, step.categories_)}
        out[name] = d
    out["feature_names_out"] = list(pre.get_feature_names_out())
    return out


def write_final_hyperparameters(dev_sel, seeds):
    obj = dict(
        selected_code=dev_sel["final"], procedure=dev_sel["final_procedure"],
        config={k: v for k, v in dev_sel["final_config"].items() if k != "purpose"},
        selection_rule=dev_sel["rule"],
        dev_inner_cv_mean_AP=dev_sel["final_dev_inner_mean_AP"],
        dev_inner_cv_sd_AP=dev_sel["final_dev_inner_sd_AP"],
        manual_best=dev_sel["manual_best"], random_best=dev_sel["random_best"],
        threshold=THRESHOLD, threshold_note="Varsayılan 0.50; eşik araması yapılmamıştır.",
        primary_seed=seeds["final_seeds"][0], stability_seeds=seeds["final_seeds"],
        early_stop_split_seed=seeds["early_stop_split"],
        training_data="Tüm development (3.295) -> %85 fit / %15 early-stop (stratified)",
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      sklearn=sklearn.__version__, platform=platform.platform()),
        **git_state(),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    with open(RESULTS_DIR / "final_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return obj


# ----------------------------------------------------------------------------- nihai eğitim
def train_final_models(df_dev, config, seeds, verbose=True):
    """Nihai konfigürasyonu 3 seed ile ve LogReg'i tüm development üzerinde eğitir, diske kaydeder."""
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = split_xy(df_dev)
    results, rows, hists = {}, [], []
    for s in seeds["final_seeds"]:
        res = train_mlp(config, X, y, seed=s, es_seed=seeds["early_stop_split"])
        results[s] = res
        torch.save(res.model.state_dict(), MODELS_DIR / f"mlp_seed{s}.pt")
        joblib.dump(res.preprocessor, MODELS_DIR / f"preprocessor_seed{s}.joblib")
        with open(MODELS_DIR / f"learned_stats_seed{s}.json", "w", encoding="utf-8") as f:
            json.dump(learned_stats(res.preprocessor), f, ensure_ascii=False, indent=1)
        rows.append(dict(seed=s, primary=(s == seeds["final_seeds"][0]), **res.summary))
        hists.append(res.history.assign(seed=s))
        if verbose:
            sm = res.summary
            print(f"seed {s}: best_epoch={sm['best_epoch']} stopped={sm['stopped_epoch']} "
                  f"best_val_loss={sm['best_val_loss']:.4f} params={sm['n_params']} "
                  f"t={sm['train_time_s']:.1f}s {sm['warnings'] or ''}")

    # LogisticRegression referansı: X_ev olarak boş bir çerçeve verilmez; model ayrıca döndürülür
    _, logreg = fit_baselines(X, y, X.head(1), seeds["torch_init"])
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "final_training_summary.csv", index=False)
    pd.concat(hists, ignore_index=True).to_csv(RESULTS_DIR / "final_training_histories.csv", index=False)
    return results, logreg, summary


# ----------------------------------------------------------------------------- kilitli tahminler
def write_locked_predictions(results, logreg, X_test, seeds):
    """Etiketsiz test tahminlerini yazar ve bir manifest (sha256, git durumu) üretir."""
    assert TARGET not in X_test.columns, "Test özelliklerinde etiket sütunu olmamalı!"
    out = pd.DataFrame({"orig_index": X_test.index})
    for s in seeds["final_seeds"]:
        p = predict_proba(results[s], X_test)
        out[f"prob_mlp_s{s}"] = p
        out[f"pred_mlp_s{s}"] = (p >= THRESHOLD).astype(int)
    p = logreg.predict_proba(X_test)[:, 1]
    out["prob_logreg"] = p
    out["pred_logreg"] = (p >= THRESHOLD).astype(int)
    assert TARGET not in out.columns
    out.to_csv(LOCKED_PATH, index=False)

    manifest = dict(file=str(LOCKED_PATH.relative_to(ROOT)).replace("\\", "/"), sha256=sha256(LOCKED_PATH),
                    n_rows=len(out), columns=list(out.columns), primary_model=f"mlp_s{seeds['final_seeds'][0]}",
                    threshold=THRESHOLD, contains_labels=False, **git_state(),
                    created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with open(RESULTS_DIR / "test_lock_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out, manifest


def check_lock_committed():
    """Bölüm B'den önce: kilitli dosya commit edilmiş ve commit'ten sonra değişmemiş olmalı."""
    rel = "results/test_predictions_locked.csv"
    code, _ = _git("ls-files", "--error-unmatch", rel)
    if code != 0:
        raise RuntimeError("Kilitli tahmin dosyası git'te yok. Önce commit edin; test etiketleri okunmayacak.")
    code, _ = _git("diff", "--quiet", "HEAD", "--", rel)
    if code != 0:
        raise RuntimeError("Kilitli tahmin dosyası commit'ten sonra değişmiş. Test etiketleri okunmayacak.")
    _, info = _git("log", "-1", "--format=%H|%ci|%s", "--", rel)
    h, date, msg = info.split("|", 2)
    with open(RESULTS_DIR / "test_lock_manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    assert man["sha256"] == sha256(LOCKED_PATH), "Manifest sha256 uyuşmuyor!"
    return dict(lock_commit=h, lock_commit_date=date, lock_commit_message=msg, sha256=man["sha256"])
