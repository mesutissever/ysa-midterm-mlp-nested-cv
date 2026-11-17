"""Kat-içi (fold-local) ön işleme.

Kural: build_preprocessor() her çağrıda YENİ ve fit EDİLMEMİŞ bir pipeline döndürür.
fit() yalnızca MLP ağırlıklarını eğiten "fit" alt kümesinde çağrılır; early-stop alt kümesi,
inner validation, outer evaluation ve test için yalnızca transform() kullanılır.

Akış:
  Sayısal (pdays hariç 8 sütun): median impute -> [yüzdelik clipping] -> scaler
  pdays                       : median impute -> scaler   (999 = "temas yok" özel kodu; kırpılmaz)
  Kategorik (10 sütun)        : most_frequent impute (yalnızca gerçek NaN için; "unknown" ayrı kategori
                                olarak korunur) -> OneHotEncoder(handle_unknown="ignore")
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

NUM_COLS = ["age", "campaign", "pdays", "previous", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
NO_CLIP_COLS = ["pdays"]                      # özel kodlu sütun: kırpma uygulanmaz
CLIP_COLS = [c for c in NUM_COLS if c not in NO_CLIP_COLS]

SCALERS = {
    "standard": StandardScaler,   # z-skoru (standardization)
    "minmax": MinMaxScaler,       # [0, 1] aralığı (normalization)
    "robust": RobustScaler,       # medyan / IQR
}

CLIP_PERCENTILES = (1.0, 99.0)


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Her sütun için alt/üst yüzdelik eşiklerini fit verisinden öğrenir ve kırpar."""

    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(scaler="standard", clip=False):
    """Fit edilmemiş ColumnTransformer döndürür.

    scaler: "standard" | "minmax" | "robust"
    clip  : True ise pdays dışındaki sayısal sütunlara 1.–99. yüzdelik kırpma uygulanır.
    """
    if scaler not in SCALERS:
        raise ValueError(f"Bilinmeyen scaler: {scaler}")

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if clip:
        num_steps.append(("clip", PercentileClipper(*CLIP_PERCENTILES)))
    num_steps.append(("scale", SCALERS[scaler]()))

    pdays_steps = [("impute", SimpleImputer(strategy="median")),
                   ("scale", SCALERS[scaler]())]

    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), CLIP_COLS),
            ("pdays", Pipeline(pdays_steps), NO_CLIP_COLS),
            ("cat", Pipeline(cat_steps), CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_logreg_preprocessor():
    """Logistic Regression referansı için sabit ön işleme kuralı."""
    return build_preprocessor(scaler="standard", clip=False)


def fit_transform_split(preprocessor, X_fit, *others):
    """preprocessor'ı YALNIZCA X_fit üzerinde fit eder, diğerlerine sadece transform uygular.

    Dönüş: (Z_fit, Z_other1, Z_other2, ...) float32 numpy dizileri.
    """
    Z_fit = preprocessor.fit_transform(X_fit).astype(np.float32)
    return (Z_fit, *[preprocessor.transform(X).astype(np.float32) for X in others])


def _leak_check():
    """Basit sızıntı kontrolü: öğrenilen istatistikler yalnızca fit alt kümesinden gelmeli."""
    from sklearn.model_selection import train_test_split
    from src.data import TARGET, get_dev_test

    df_dev, _ = get_dev_test()                     # test kullanılmıyor
    X, y = df_dev.drop(columns=TARGET), df_dev[TARGET]
    X_fit, X_other, y_fit, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    for scaler in SCALERS:
        for clip in (False, True):
            pre = build_preprocessor(scaler, clip)
            Z_fit, Z_other = fit_transform_split(pre, X_fit, X_other)
            print(f"scaler={scaler:8s} clip={clip!s:5s} -> boyut fit={Z_fit.shape}, other={Z_other.shape}")

    # 1) StandardScaler ortalaması = fit alt kümesinin ortalaması (tüm verinin değil)
    pre = build_preprocessor("standard", clip=False)
    fit_transform_split(pre, X_fit, X_other)
    mean_learned = pre.named_transformers_["num"].named_steps["scale"].mean_
    assert np.allclose(mean_learned, X_fit[CLIP_COLS].mean().values)
    assert not np.allclose(mean_learned, X[CLIP_COLS].mean().values)
    print("OK: StandardScaler istatistikleri yalnızca fit alt kümesinden öğrenildi.")

    # 2) Clipping eşikleri fit alt kümesinin yüzdelikleri; pdays kırpılmıyor
    pre = build_preprocessor("standard", clip=True)
    fit_transform_split(pre, X_fit, X_other)
    clipper = pre.named_transformers_["num"].named_steps["clip"]
    assert np.allclose(clipper.upper_, np.percentile(X_fit[CLIP_COLS].values, 99, axis=0))
    assert "pdays" not in CLIP_COLS
    print("OK: Kırpma eşikleri fit alt kümesinden öğrenildi; pdays kırpılmıyor.")
    print("    Eşikler:", dict(zip(CLIP_COLS, zip(clipper.lower_.round(3), clipper.upper_.round(3)))))

    # 3) Fit alt kümesinde görülmemiş kategori hata vermez (sıfır vektör)
    X_new = X_other.head(1).copy()
    X_new["job"] = "astronaut"
    z = pre.transform(X_new)
    names = pre.get_feature_names_out()
    job_cols = [i for i, n in enumerate(names) if n.startswith("job_")]
    assert np.all(z[0, job_cols] == 0)
    print("OK: Görülmemiş kategori -> tüm job_* sütunları 0 (handle_unknown='ignore').")
    print(f"Girdi boyutu: {len(names)} (sayısal {len(NUM_COLS)} + one-hot {len(names) - len(NUM_COLS)})")


if __name__ == "__main__":
    _leak_check()
