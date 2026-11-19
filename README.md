# MLP ile Sınıflandırma: EDA, Kat-İçi Ön İşleme, Manuel/Otomatik HP Araması ve Nested CV

Yapay Sinir Ağları (Yüksek Lisans) ara sınav projesi. UCI Bank Marketing (`bank-additional.csv`) verisinde, görüşme öncesi bilinen özelliklerle vadeli mevduat aboneliğini tahmin eden PyTorch MLP modelleri; veri sızıntısını önleyen 3×3 iç içe çapraz doğrulama ile 6 manuel (M1–M6) ve 6 Random Search (R1–R6) konfigürasyonunun karşılaştırılması ve kilitli bağımsız test.

📄 **Rapor:** [`report.pdf`](report.pdf) (kaynak: [`report/rapor.md`](report/rapor.md))

## Temel sonuçlar

| | AP | ROC-AUC | F1 |
|---|---|---|---|
| Dış CV — manuel arama (Mean ± SD, n=3) | 0,429 ± 0,055 | 0,780 ± 0,026 | 0,311 ± 0,041 |
| Dış CV — otomatik arama | 0,418 ± 0,039 | 0,773 ± 0,021 | 0,335 ± 0,049 |
| Dış CV — LogisticRegression | 0,420 ± 0,028 | 0,774 ± 0,015 | 0,365 ± 0,047 |
| Dış CV — Dummy (prior) | 0,110 | 0,500 | 0,000 |
| **Bağımsız test — nihai MLP (M2, seed 31046)** [%95 bootstrap GA] | **0,406** [0,313; 0,519] | **0,744** [0,682; 0,807] | **0,320** [0,215; 0,431] |
| Bağımsız test — LogisticRegression | 0,390 [0,300; 0,509] | 0,737 [0,675; 0,802] | 0,333 [0,222; 0,439] |

Nihai konfigürasyon (tüm development iç CV'si ile seçildi): **M2** — Adam, gizli katman (32,), ReLU, LR = 0,001, weight decay = 1e-4, batch 64, StandardScaler, 2.017 parametre. MLP ile LogisticRegression arasındaki test farkı istatistiksel olarak ayırt edilememiştir (ΔAP = +0,016; %95 GA −0,014 ile +0,041).

## Klasör yapısı

```
├── 01_eda.ipynb              Development-only EDA ve grafikler
├── 02_nested_cv.ipynb        Kat-içi ön işleme, manuel/otomatik arama, 3×3 nested CV
├── 03_final_test.ipynb       Nihai eğitim, kilitli tahminler, test metrikleri, bootstrap, grafikler
├── src/
│   ├── data.py               Veri okuma/indirme, kontroller, stratified %80/%20 bölme
│   ├── preprocessing.py      ColumnTransformer (imputation, kırpma, ölçekleme, one-hot) + sızıntı testi
│   ├── model.py              PyTorch MLP
│   ├── train.py              Eğitim döngüsü (early stopping, ReduceLROnPlateau)
│   ├── configs.py            M1–M6 ve Random Search (R1–R6) adaylarının üretimi
│   ├── nested_cv.py          Nested CV, referans modeller, metrikler
│   ├── final.py              Nihai eğitim, model kaydı, kilitli tahminler, kilit doğrulaması
│   ├── evaluate.py           Test metrikleri, bootstrap GA, seed kararlılığı, hata analizi
│   └── plots.py              MLP mimari şeması
├── configs/                  seeds, eğitim ayarları, arama uzayı, manuel ve otomatik adaylar (JSON)
├── results/                  split, iç/dış CV sonuçları, tahminler, nihai metrikler (CSV/JSON)
├── figures/                  Raporda kullanılan tüm grafikler
├── models/                   Nihai modeller (3 seed), ön işleme nesneleri, LogisticRegression
├── report/                   Rapor kaynağı (Markdown) ve PDF üretim betiği
├── report.pdf                Teknik rapor
└── requirements.txt
```

## Veri

- Kaynak: Moro, S., Rita, P. ve Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/222/bank+marketing — DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306)
- Yalnızca `bank-additional.csv` (4.119 × 21) kullanılır. Ham veri depoya yüklenmemiştir (`data/` git'te yok sayılır).
- **Otomatik indirme:** `data/bank-additional.csv` yoksa ilk okumada UCI'deki `bank+marketing.zip` → `bank-additional.zip` → `bank-additional/bank-additional.csv` dosyası indirilir ve sha256 ile doğrulanır (`7e59cf65…5df50`).
- Elle indirmek için: zip'i açıp `bank-additional.csv` dosyasını `data/` klasörüne koyun.

## Kurulum

```bash
git clone https://github.com/mesutissever/ysa-midterm-mlp-nested-cv.git
cd ysa-midterm-mlp-nested-cv
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Google Colab:** Her notebook'un ilk kod hücresindeki `git clone`, `%cd` ve `pip install` satırlarının başındaki `#` kaldırılarak çalıştırılır.

## Çalıştırma sırası

| Adım | Komut / dosya | Süre (CPU) | Çıktılar |
|---|---|---|---|
| 1 | `python -m src.data` | < 5 sn | `results/dataset_split.csv` (stratified %80/%20, seed 31041) |
| 2 | `01_eda.ipynb` | < 30 sn | `figures/eda_*.png`, `results/eda_*.csv` |
| 3 | `02_nested_cv.ipynb` | ≈ 3 dk | iç/dış CV sonuçları, `results/dev_selection.json`, eğitim geçmişleri |
| 4 | `03_final_test.ipynb` | ≈ 15 sn | nihai modeller, kilitli tahminler, test metrikleri, bootstrap, Şekil 3/6/7 |

Notlar:
- Adaylar (`configs/manual_configs.json`, `configs/random_configs.json`) nested CV'den önce üretilip commit edilmiştir. `python -m src.configs` mevcut `random_configs.json` dosyasının üzerine yazmayı reddeder.
- `03_final_test.ipynb` baştan sona (Run All) çalıştırılabilir. Kilit commit'i mevcut olduğundan Bölüm A yeniden eğitim yapmaz ve kilitli dosyanın üzerine yazmaz; kayıtlı modelleri diskten okur. Bölüm B, kilitli dosya commit edilmemişse veya değiştirilmişse test etiketlerini okumayı reddeder.
- Rapor PDF'i: `python report/build_pdf.py` (ek paket: `pip install markdown`; Chrome veya Edge gerekir).

## Seed politikası

Temel seed = öğrenci numarasının son beş hanesi (29015) + 2026 = **31041**. Alt seed'ler sabit ofsetle türetilmiştir (`configs/seeds.json`):

| Amaç | Seed |
|---|---|
| Development/test bölünmesi | 31041 |
| Outer CV / Inner CV (StratifiedKFold, 3 kat) | 31042 / 31043 |
| Early-stop alt kümesi (%15, stratified) | 31044 |
| Random Search örneklemesi | 31045 |
| PyTorch ağırlık başlangıcı (ana) | 31046 |
| Nihai seed kararlılığı | 31046, 31146, 31246 |
| Bootstrap | 31047 |

Her eğitimden önce Python, NumPy ve PyTorch seed'leri ayarlanır ve `torch.use_deterministic_algorithms(True)` etkinleştirilir. Farklı donanım veya kitaplık sürümlerinde küçük sayısal farklar oluşabilir.

## Kilitli test protokolü

| Commit | İçerik |
|---|---|
| `8a81da2` | Nihai eğitim ve kilitli tahmin kodu |
| `cb19a1c` | Etiketsiz `results/test_predictions_locked.csv` + `results/test_lock_manifest.json` (sha256) + modeller |
| `50b5b02` | Test metrikleri — yalnızca kilit commit'i doğrulandıktan sonra |

Test etiketleri kamuya açık olduğundan bu işlem kör test garantisi değil, zaman sıralı bir deney kaydıdır.

## Ortam

- Windows 11 Pro for Workstations (10.0.26200), Intel64 CPU, GPU kullanılmadı
- Python 3.13.1 · torch 2.15.0 · scikit-learn 1.9.1 · pandas 3.1.1 · numpy 2.5.4 · matplotlib 3.11.2 · seaborn 0.13.2
