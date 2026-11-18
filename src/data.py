"""Veri okuma, temel kontroller ve stratified development/test bölmesi.

Kurallar (ödev Bölüm 2-3):
- Yalnızca bank-additional.csv (4119 x 21) kullanılır.
- y: "yes" -> 1 (pozitif), "no" -> 0.
- "duration" görüşme öncesi bilinmediği için modelden çıkarılır.
- "unknown" kategorileri NaN'a çevrilmez; pdays=999 silinmez/değiştirilmez.
"""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-additional.csv"
RESULTS_DIR = ROOT / "results"
TARGET = "y"
DROP_COLS = ["duration"]

EXPECTED_SHAPE = (4119, 21)
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_SHA256 = "7e59cf650004d65d1c9d6b08553bad2ee9a9ad70d594f536e3c584ee6ed5df50"


def load_seeds(path=ROOT / "configs" / "seeds.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_data(path=DATA_PATH):
    """Veri yoksa UCI'den indirir (bank+marketing.zip -> bank-additional.zip -> bank-additional.csv)
    ve sha256 ile kullanılan dosyanın aynısı olduğunu doğrular."""
    path = Path(path)
    if not path.exists():
        print(f"{path.name} bulunamadı, UCI'den indiriliyor: {UCI_URL}")
        with urllib.request.urlopen(UCI_URL, timeout=120) as r:
            outer = zipfile.ZipFile(io.BytesIO(r.read()))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip")))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(inner.read("bank-additional/bank-additional.csv"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, f"Veri dosyası beklenen sürüm değil (sha256={digest})"
    return path


def load_raw(path=DATA_PATH):
    """Ham CSV'yi okur ve yapısal kontrolleri yapar (hiçbir değişiklik yapmaz)."""
    df = pd.read_csv(ensure_data(path), sep=";")
    assert df.shape == EXPECTED_SHAPE, f"Beklenen {EXPECTED_SHAPE}, gelen {df.shape}"
    labels = set(df[TARGET].unique())
    assert labels == {"yes", "no"}, f"Beklenmeyen hedef etiketleri: {labels}"
    return df


def clean(df_raw):
    """Hedefi 0/1'e çevirir ve 'duration'ı çıkarır. 'unknown' ve pdays=999 korunur."""
    df = df_raw.copy()
    df[TARGET] = (df[TARGET] == "yes").astype(int)
    df = df.drop(columns=DROP_COLS)
    return df


def load_data(path=DATA_PATH):
    """Modelleme için hazır veri: (4119, 20) -> 19 özellik + y."""
    return clean(load_raw(path))


def check_duplicates(df_raw, df_clean):
    """Birebir kopya satırları duration silinmeden önce ve sonra sayar."""
    feats = [c for c in df_clean.columns if c != TARGET]
    info = {
        "dup_rows_raw": int(df_raw.duplicated().sum()),
        "dup_rows_after_drop_duration": int(df_clean.duplicated().sum()),
        # aynı özellikler, farklı etiket -> çelişkili örnek
        "dup_features_only": int(df_clean.duplicated(subset=feats).sum()),
        "conflicting_label_groups": int(
            df_clean.groupby(feats, dropna=False)[TARGET].nunique().gt(1).sum()
        ),
    }
    for k, v in info.items():
        print(f"{k:32s}: {v}")
    return info


def make_split(df, seed, test_size=0.20, save=True):
    """Stratified %80 development / %20 test bölmesi; orijinal indeksleri kaydeder."""
    dev_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    split = pd.DataFrame({"orig_index": df.index, TARGET: df[TARGET].values})
    split["split"] = "development"
    split.loc[split["orig_index"].isin(test_idx), "split"] = "test"

    summary = (
        split.groupby("split")[TARGET]
        .agg(n="size", n_pos="sum", pos_rate="mean")
        .reset_index()
    )
    print(summary.to_string(index=False))

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        split.to_csv(RESULTS_DIR / "dataset_split.csv", index=False)
        summary.to_csv(RESULTS_DIR / "dataset_split_summary.csv", index=False)
    return dev_idx.sort_values(), test_idx.sort_values(), summary


def get_dev_test(path=DATA_PATH, split_path=RESULTS_DIR / "dataset_split.csv"):
    """Kaydedilmiş bölmeyi okuyup (df_dev, df_test) döndürür. Sonraki notebook'lar bunu kullanır."""
    df = load_data(path)
    split = pd.read_csv(split_path)
    dev_idx = split.loc[split["split"] == "development", "orig_index"].values
    test_idx = split.loc[split["split"] == "test", "orig_index"].values
    return df.loc[dev_idx], df.loc[test_idx]


def get_test_features(path=DATA_PATH, split_path=RESULTS_DIR / "dataset_split.csv"):
    """Test kümesinin YALNIZCA özelliklerini döndürür (etiket sütunu çıkarılmış).
    Kilitli tahmin aşamasında kullanılır; test etiketleri bu fonksiyonla okunmaz."""
    _, df_test = get_dev_test(path, split_path)
    return df_test.drop(columns=TARGET)


if __name__ == "__main__":
    seeds = load_seeds()
    raw = load_raw()
    df = clean(raw)
    print("Shape (temiz):", df.shape)
    check_duplicates(raw, df)
    make_split(df, seed=seeds["split"])
