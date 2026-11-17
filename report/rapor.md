# Banka Pazarlama Verisinde MLP ile İkili Sınıflandırma: Kat-İçi Ön İşleme, Manuel ve Otomatik Hiperparametre Araması ve İç İçe Çapraz Doğrulama

**Ders:** Yapay Sinir Ağları (Yüksek Lisans) — Ara Sınav Projesi
**Öğrenci:** [Ad Soyad], [Öğrenci No]
**GitHub:** https://github.com/mesutissever/ysa-midterm-mlp-nested-cv
**Tarih:** [Teslim tarihi]

> **Taslak notu (teslimden önce silinecek):** `[SONUÇ]` ile işaretli yerler ilgili aşama çalıştırıldıktan sonra `results/` altındaki CSV/JSON dosyalarından doldurulacaktır. Hiçbir sayı elle yazılmayacaktır.

---

## Özet

[SONUÇ — en son yazılacak. 150–200 kelime: problem, veri, yöntem (fold-local ön işleme, 3×3 nested CV, 6 manuel + 6 Random Search adayı, AP birincil metrik), outer CV Mean ± SD, bağımsız test AP/ROC-AUC ve %95 bootstrap CI, temel çıkarım.]

**Araştırma sorusu:** Görüşme öncesi bilinen müşteri ve makroekonomik özelliklerle vadeli mevduat aboneliğini tahmin eden bir MLP'nin performansı; ön işleme (ölçekleyici, aykırı değer kırpma), mimari, optimizer ve öğrenme oranı planlaması tercihlerine ne kadar duyarlıdır ve hipotez temelli manuel arama ile eşit bütçeli rastgele arama, sızıntısız iç içe çapraz doğrulama altında nasıl karşılaştırılır?

---

## 1. Giriş

Çok Katmanlı Algılayıcı (Multi-Layer Perceptron, MLP), doğrusal dönüşümlerin doğrusal olmayan aktivasyon fonksiyonlarıyla art arda uygulandığı ileri beslemeli bir yapay sinir ağıdır. Yeterli gizli birim sayısıyla sürekli fonksiyonlara keyfî yakınlıkta yaklaşabilmesi MLP'yi tablo verilerinde güçlü bir aday yapar; ancak bu esneklik, modelin başarısını büyük ölçüde hiperparametre seçimlerine (mimari, optimizer, öğrenme oranı, düzenlileştirme, yığın boyutu) ve girdi ön işlemesine bağımlı kılar. Küçük ve dengesiz veri setlerinde bu duyarlılık daha belirgindir: aşırı parametreli bir ağ kolayca aşırı öğrenebilir, uygunsuz ölçekleme veya öğrenme oranı ise yakınsamayı yavaşlatabilir.

Hiperparametre optimizasyonu genellikle iki yolla yapılır: alan bilgisine ve hipotezlere dayalı **manuel arama** ve arama uzayından aday örnekleyen **otomatik arama** (ör. Random Search). Her iki yaklaşımın da adil karşılaştırılabilmesi, aynı veri bölünmeleri, aynı değerlendirme metriği ve aynı aday bütçesiyle çalıştırılmalarını gerektirir. Ayrıca hiperparametre seçimi ile performans tahmini aynı veride yapıldığında elde edilen skor iyimser biçimde yanlı olur. **İç içe çapraz doğrulama (nested cross-validation)**, seçimi iç döngüde, performans tahminini dış döngüde yaparak bu yanlılığı azaltır.

Bu çalışmada kullanılan Bank Marketing verisinde pozitif sınıf oranı yaklaşık %11'dir. Böyle **dengesiz** bir problemde doğruluk (accuracy) yanıltıcıdır: tüm örnekleri negatif tahmin eden bir model bile yaklaşık %89 doğruluğa ulaşır. Bu nedenle birincil metrik olarak pozitif sınıfa odaklanan **Average Precision (AP)** seçilmiş; ROC-AUC, F1, duyarlılık (sensitivity), özgüllük (specificity) ve dengelenmiş doğruluk (balanced accuracy) tamamlayıcı metrikler olarak raporlanmıştır.

Çalışmanın amaçları şunlardır: (i) veri sızıntısını önleyen, tekrarlanabilir bir deney akışı kurmak; (ii) altı hipotez temelli manuel ve altı rastgele aday konfigürasyonu aynı 3×3 nested CV düzeninde karşılaştırmak; (iii) seçilen modelin genelleme başarısını yalnızca bir kez kullanılan bağımsız test kümesinde, güven aralıkları ve seed kararlılığıyla birlikte ölçmek.

---

## 2. Veri Seti ve Keşifsel Veri Analizi

### 2.1 Veri seti ve kullanım kuralları

UCI Machine Learning Repository'deki Bank Marketing veri setinin yalnızca `bank-additional.csv` dosyası kullanılmıştır (Moro vd., 2014). Dosya 4.119 gözlem, 20 girdi özelliği ve ikili hedef değişken `y` içerir. Daha büyük `bank-additional-full.csv` sürümü kullanılmamıştır. Hedef değişken "yes" = 1 (pozitif), "no" = 0 (negatif) olarak kodlanmıştır.

Modelleme öncesinde, performans incelenmeden önce aşağıdaki kurallar sabitlenmiştir:

- **`duration` çıkarıldı.** Görüşme süresi, görüşme gerçekleşmeden bilinemez ve hedefle doğrudan ilişkilidir (süre 0 ise sonuç zorunlu olarak "no"dur). Gerçekçi bir tahmin senaryosunda kullanılamayacağından tüm modellerden çıkarılmıştır. Böylece 19 girdi özelliği kalmıştır: 9 sayısal, 10 kategorik.
- **"unknown" değerleri NaN kabul edilmedi.** Kategorik değişkenlerdeki "unknown" kodu ayrı bir kategori olarak korunmuştur; bilginin yokluğu da bilgi taşıyabilir (ör. `default` değişkeninde).
- **`pdays = 999` korundu.** Bu değer "müşteriyle daha önce temas kurulmamış" anlamına gelen özel bir koddur; sıradan aykırı değer olarak silinmemiştir.

### 2.2 Seed politikası ve veri bölünmesi

Temel seed, öğrenci numarasının son beş hanesine (29015) 2026 eklenerek **31041** olarak belirlenmiştir. Alt seed'ler temel seed'e sabit ofset eklenerek deterministik biçimde türetilmiş ve deneylerden önce `configs/seeds.json` dosyasında kayıt altına alınmıştır (Tablo 0).

**Tablo 0.** Seed politikası.

| Amaç | Seed |
|---|---|
| Development/test bölünmesi | 31041 |
| Outer CV (StratifiedKFold) | 31042 |
| Inner CV (StratifiedKFold) | 31043 |
| Early-stop alt kümesi ayrımı | 31044 |
| Random Search aday örneklemesi | 31045 |
| PyTorch ağırlık başlangıcı (ana) | 31046 |
| Seed kararlılığı için nihai seed'ler | 31046, 31146, 31246 |
| Bootstrap | 31047 |

Veri, hedef değişkene göre tabakalı (stratified) olarak %80 development ve %20 bağımsız test kümesine ayrılmıştır. Test kümesi EDA, ön işleme fit işlemleri, hiperparametre seçimi ve eşik belirlemede hiç kullanılmamıştır. Orijinal satır indeksleri `results/dataset_split.csv` dosyasına kaydedilmiştir.

**Tablo 1a.** Development/test bölünmesi ve sınıf dağılımı.

| Küme | n | Pozitif (y=1) | Pozitif oranı |
|---|---|---|---|
| Development | 3.295 | 361 | %10,96 |
| Bağımsız test | 824 | 90 | %10,92 |
| **Toplam** | **4.119** | **451** | **%10,95** |

**Kopya kayıtlar.** Ham veride birebir kopya satır bulunmamaktadır. `duration` çıkarıldıktan sonra 19 satır, başka bir satırla tüm özellikleri ve etiketi bakımından aynı hâle gelmiştir (20 kopya grubu; bunlardan birinde özellikler aynı, etiketler farklıdır). Bu satırlar ham veride farklı görüşmeleri temsil ettiğinden silinmemiştir. Kopya gruplarından 6'sı hem development hem test kümesinde örnek içermektedir. Test kümesinin büyüklüğü (824) düşünüldüğünde bu durumun etkisinin sınırlı olması beklenir; yine de bir sınırlama olarak not edilmiştir.

**Kişi-bağımsız genelleme sınırlaması.** Veride müşteri kimliği bulunmadığından aynı müşteriye ait birden fazla görüşme farklı kümelere düşebilir. Ayrıca veri zaman sırasına göre değil rastgele bölünmüştür. Bu nedenle elde edilen test performansı gelecekteki kampanyalara veya yeni müşterilere genellemeyi kanıtlamaz.

### 2.3 Tanımlayıcı istatistikler

[SONUÇ — `01_eda.ipynb`'den. Yalnızca development verisi (n = 3.295).]

**Tablo 1b.** Sayısal özellikler: ortalama, medyan, SD, IQR, min–maks. [SONUÇ]

**Tablo 1c.** Kategorik özelliklerde "unknown" sayıları/oranları ve `pdays = 999` oranı. [SONUÇ]

**Şekil 2a.** Sınıf dağılımı (development). `figures/eda_class_dist.png` [SONUÇ]

**Şekil 2b.** Seçilmiş sayısal özelliklerin histogram ve boxplot'ları. [SONUÇ]

**Şekil 2c.** Sayısal özellikler korelasyon matrisi. [SONUÇ]

[Yorum: çarpık dağılımlar (`campaign`, `previous`), `pdays` özel kodu, makroekonomik değişkenler arasındaki yüksek korelasyon, IQR ile işaretlenen aykırı değer sayıları. Hiçbir özellik EDA'ya dayanarak elenmemiştir; böyle bir karar model seçimi sayılacağından yalnızca iç CV içinde verilebilir.]

### 2.4 Deney hipotezleri

EDA bulgularına dayanarak, dış CV sonuçları görülmeden önce şu hipotezler belirlenmiştir: [EDA sonrası kesinleştirilecek]

- **H1 (ölçekleme / aykırı değer):** Sağa çarpık sayısal özellikler nedeniyle, eğitim katında öğrenilmiş yüzdelik kırpma veya RobustScaler, StandardScaler'a kıyasla daha yüksek iç CV AP değeri sağlar. MinMaxScaler aykırı değerlere duyarlı olduğundan StandardScaler'dan daha düşük AP verir (M1 ve M4 karşılaştırması).
- **H2 (optimizer):** Aynı öğrenme oranında (0,001) SGD+Momentum, uyarlamalı yöntemlere (Adam, AdamW) göre daha yavaş yakınsar ve 150 epoch sınırında daha düşük AP'ye ulaşır (M1, M2, M3 karşılaştırması). Daha geniş öğrenme oranı aralığına sahip rastgele aramada bu fark azalır.
- **H3 (mimari):** Yaklaşık 3.300 örnekli küçük veri setinde daha büyük mimari (64,32), tek gizli katmanlı (32,) mimariye göre anlamlı bir AP kazancı sağlamaz (M1 ve M5 karşılaştırması).
- **H4 (öğrenme oranı planlaması):** ReduceLROnPlateau kapatıldığında (M6) erken durdurma daha erken tetiklenir ve AP bir miktar düşer.

---

## 3. Yöntem

### 3.1 Deneysel akış ve veri sızıntısının önlenmesi

Deney akışı Şekil 1'de özetlenmiştir. Bağımsız test kümesi ayrıldıktan sonra tüm geliştirme kararları yalnızca development verisi üzerinde, iç içe çapraz doğrulamayla verilmiştir.

**Şekil 1.** Veri bölünmesi ve nested CV akış şeması. `figures/flow_nested_cv.png` [OLUŞTURULACAK]

1. Development verisi, `StratifiedKFold(n_splits=3, shuffle=True, random_state=31042)` ile üç **dış kata** ayrılır.
2. Her dış katın eğitim bölümü (outer-train), `StratifiedKFold(n_splits=3, shuffle=True, random_state=31043)` ile üç **iç kata** ayrılır.
3. Her iç katın eğitim bölümü ayrıca ikiye bölünür: **%85 "fit" alt kümesi** ön işleme parametrelerini öğrenir ve MLP ağırlıklarını günceller; **%15 tabakalı "early-stop monitor" alt kümesi** yalnızca erken durdurma ve öğrenme oranı planlayıcısını yönetir.
4. Her aday konfigürasyonun AP değeri **iç doğrulama katında** hesaplanır; aday seçimi üç iç katın ortalama AP'sine göre yapılır.
5. Manuel (M1–M6) ve otomatik (R1–R6) aramalar aynı iç katlarda **ayrı ayrı** yürütülür. Her aramanın kazanan adayı outer-train üzerinde yeniden eğitilir ve dış değerlendirme katında **yalnızca bir kez** ölçülür.
6. Dış kat skorları arama prosedürlerinin genelleme performansını tahmin etmek için raporlanır; konfigürasyonları değiştirmek veya iki stratejiden birini seçmek için **kullanılmaz**.
7. Nihai seçim için tam development verisinde iç CV yeniden çalıştırılır; development iç CV ortalama AP değeri en yüksek olan konfigürasyon seçilir.
8. Nihai model üç seed ile eğitilir. Test tahminleri etiketsiz olarak `test_predictions_locked.csv` dosyasına yazılır ve ayrı bir commit ile GitHub'a yüklenir. Test metrikleri yalnızca bu commit'ten sonra hesaplanır.

**Tablo 2.** Eğitim bütçesi.

| Hesap | Eğitim sayısı |
|---|---|
| 3 dış × 3 iç × (6 manuel + 6 otomatik) | 108 |
| Dış katlarda seçilen iki yöntemin yeniden eğitimi | 6 |
| Tam development üzerinde 3 iç × 12 aday | 36 |
| Nihai model, 3 seed | 3 |

### 3.2 Kat-içi ön işleme

Ön işleme, scikit-learn `Pipeline` ve `ColumnTransformer` ile tanımlanmıştır. Tüm `fit` işlemleri yalnızca ilgili MLP ağırlık eğitim alt kümesinde ("fit" alt kümesi) yapılır; early-stop alt kümesine, iç doğrulama katına, dış değerlendirme katına ve teste yalnızca `transform` uygulanır. Öğrenilen parametreler şunlardır: medyan değerleri, kategori sözlüğü, ölçekleyici istatistikleri ve kırpma eşikleri.

- **Sayısal özellikler:** medyan imputation (development verisinde sayısal eksik değer bulunmamaktadır; adım, pipeline'ın genelliği için korunmuştur) → isteğe bağlı yüzdelik kırpma (eşikler fit alt kümesinin 1. ve 99. yüzdeliklerinden öğrenilir) → ölçekleyici.
- **`pdays` kırpılmaz.** Değerlerin yaklaşık %96'sı "daha önce temas yok" anlamındaki 999 özel kodudur; bu nedenle 99. yüzdelik 999'a, 1. yüzdelik küçük bir değere denk gelir ve kırpma yalnızca gerçek gün değerlerini (0–2) bozar. Bu karar hedef değişkene bakılmadan, değişkenin kodlama yapısına dayanılarak verilmiştir. Kırpma seçeneği açıkken yalnızca diğer sekiz sayısal özelliğe uygulanır.
- **"unknown" ayrı kategori olarak sabitlenmiştir.** Veri sözlüğüne göre "unknown" bilginin alınamadığını kodlar ve bu durumun kendisi bilgi taşıyabilir. Karar, hedefle ilişkiye bakılarak değil önceden verilmiştir; alternatif (kat-içi en sık sınıfla doldurma) arama uzayı bütçesini aşmamak için değerlendirilmemiş ve sınırlama olarak not edilmiştir. Kategorik adımda yer alan en sık sınıf imputation'ı yalnızca gerçek NaN değerler için güvenlik amacıyla bulunur.
- **Ölçekleyiciler:** **Standardization** (`StandardScaler`) her özelliği fit alt kümesinin ortalamasını çıkarıp standart sapmasına bölerek z-skoruna dönüştürür, değer aralığını sınırlamaz. **Normalization** (`MinMaxScaler`) değerleri fit alt kümesindeki min–maks aralığına göre [0, 1] aralığına taşır ve aykırı değerlere duyarlıdır. `RobustScaler` medyanı çıkarıp IQR'ye böler; aykırı değerlere karşı daha dayanıklıdır.
- **Kategorik özellikler:** "unknown" ayrı kategori olarak korunur ve `OneHotEncoder(handle_unknown="ignore")` ile kodlanır. Fit alt kümesinde görülmemiş bir kategori diğer kümelerde sıfır vektörüyle temsil edilir.

One-hot kodlamadan sonra girdi boyutu, fit alt kümesinde gözlenen kategori sayısına bağlıdır. Tüm development verisi için bu boyut 9 sayısal + 52 one-hot = **61**'dir; nadir kategoriler (ör. `education = illiterate`) bir alt kümede hiç bulunmazsa boyut 60'a düşebilir.

**Sızıntı kontrolü.** `src/preprocessing.py` içindeki otomatik kontrol, (i) ölçekleyici istatistiklerinin yalnızca fit alt kümesinin istatistiklerine eşit olduğunu, (ii) kırpma eşiklerinin fit alt kümesinin yüzdeliklerinden öğrenildiğini ve `pdays`'in kırpılmadığını, (iii) fit alt kümesinde görülmemiş bir kategorinin hata vermeden sıfır vektörle temsil edildiğini doğrulamaktadır.

### 3.3 MLP mimarisi ve eğitim

MLP'ler PyTorch ile kurulmuştur. Gizli katmanlar `Linear → aktivasyon (ReLU veya tanh) → Dropout` bloklarından oluşur. Çıkış katmanı tek bir logit üretir. Kayıp fonksiyonu `BCEWithLogitsLoss`'tur; olasılık tahmini için logit'e sigmoid uygulanır. Varsayılan sınıflandırma eşiği 0,50'dir.

**Öğrenilebilir parametre sayısı.** Girdi boyutu *d* ve gizli katman genişlikleri *h₁, h₂, …* olmak üzere parametre sayısı Σ (önceki katman × sonraki katman + sonraki katman) formülüyle hesaplanır. *d* = 61 için:
- (32,) mimarisi (M1): 61·32 + 32 + 32·1 + 1 = **2.017** parametre.
- (64,32) mimarisi (M5): 61·64 + 64 + 64·32 + 32 + 32·1 + 1 = **6.081** parametre.

Kesin değerler her eğitimde koddan hesaplanıp kaydedilmektedir.

**Şekil 3.** MLP mimari şeması ve parametre sayısı. [OLUŞTURULACAK]

**Tablo 3.** Eğitim ayarları.

| Bileşen | Ayar |
|---|---|
| Maksimum epoch | 150 (gerçek en iyi ve durulan epoch kaydedilir) |
| Optimizer | SGD + Momentum (0,9), Adam, AdamW |
| Erken durdurma | Aktif; patience = 15; en iyi ağırlıklar geri yüklenir |
| Early-stop monitor | İç eğitim bölümünden %15 tabakalı alt doğrulama |
| LR planlayıcı | ReduceLROnPlateau (mode=min, factor=0,5, patience=5, min_lr=1e-5) |
| İzlenen metrik | Early-stop alt kümesinde BCE kaybı |
| HP seçim metriği | İç CV `average_precision_score` (AP) |

**Optimizer'lar.** SGD + Momentum (0,9), tek bir global öğrenme oranıyla gradyan yönünde ilerler ve önceki adımların ivmesini kullanır; öğrenme oranına en duyarlı yöntemdir. Adam, gradyanların birinci ve ikinci momentlerini kullanarak her parametre için uyarlamalı adım büyüklüğü hesaplar; weight decay L2 cezası olarak gradyana eklendiğinden uyarlamalı ölçeklemeden etkilenir. AdamW aynı uyarlamalı güncellemeyi kullanır, ancak weight decay'i gradyandan ayrık (decoupled) biçimde doğrudan ağırlıklara uygular (Loshchilov ve Hutter, 2019). M1 (AdamW) ile M2 (Adam) karşılaştırması bu farkı ölçer.

**Diğer ayrıntılar.** Ağırlıklar PyTorch'un varsayılan başlangıcıyla (Kaiming-uniform) ilklendirilmiştir. Erken durdurmada iyileşme eşiği `min_delta = 0`'dır; planlayıcı PyTorch'un varsayılan eşiğini kullanır. Sınıf ağırlıklandırması (class weight) M1–M6 tanımlarında bulunmadığından ve tek-faktör karşılaştırmayı bozacağından uygulanmamıştır. Fit/early-stop ayrımı tüm adaylarda aynı seed (31044) ile yapılır; böylece adaylar aynı katta aynı alt kümeleri görür. Tüm eğitimler CPU üzerinde yapılmıştır. Early stopping tetiklenmeden 150 epoch'a ulaşan eğitimler "yakınsama uyarısı", sonlu olmayan kayıp üreten eğitimler "başarısız" olarak kaydedilir ve sonuçlardan çıkarılmaz.

Planlayıcı her epoch sonunda early-stop alt kümesindeki doğrulama kaybıyla güncellenir. Erken durdurma ve öğrenme oranı düşürme aynı alt kümeyi izler; iç CV doğrulama katı bu amaçla kullanılmaz. Erken durdurmada AP yerine BCE kaybı kullanılmıştır. BCE kaybı epoch'tan epoch'a daha düzgün değişir ve olasılık kalibrasyonunu da yansıtır. AP ise hiperparametre seçiminin birincil metriği olarak korunmuştur. Her eğitimde eğitim kaybı, iç doğrulama kaybı, epoch bazlı öğrenme oranı, en iyi epoch, süre ve uyarılar CSV olarak kaydedilir.

Tekrarlanabilirlik için her eğitimden önce Python, NumPy ve PyTorch seed'leri ayarlanır ve deterministik algoritmalar etkinleştirilir.

### 3.4 Referans modeller

- **DummyClassifier** (`strategy="prior"`): sınıf önceliğini tahmin eder. Dengesiz yapıya karşı alt sınır görevi görür.
- **LogisticRegression**: sabit ön işleme ile (StandardScaler + OneHotEncoder) doğrusal referans modeldir.

Referans modeller dış katlarda yalnızca outer-train üzerinde eğitilip dış değerlendirme katında ölçülmüş, ayrıca bağımsız testte nihai MLP ile karşılaştırılmıştır. Hiperparametre arama bütçesine dahil değildirler.

### 3.5 Değerlendirme metrikleri

Birincil metrik **AP**'dir (`average_precision_score`). AP, precision–recall eğrisinin adım fonksiyonu toplamıdır ve trapez yöntemiyle hesaplanan PR-AUC ile aynı isim altında sunulmamıştır. Tamamlayıcı metrikler ROC-AUC, Accuracy, Precision, Sensitivity (Recall), Specificity, F1, Balanced Accuracy ve eğitim süresidir. Dış CV sonuçları üç kat üzerinden "ortalama ± örnek standart sapma" (Mean ± SD, n = 3) olarak verilmiştir. Katlar bağımsız tekrarlar olmadığından SD bir güven aralığı gibi yorumlanmamalıdır. Bağımsız test için ROC-AUC, AP ve F1 metriklerine 1.000 tekrarlı yüzdelik bootstrap ile %95 güven aralığı hesaplanmıştır; her yeniden örneklemede iki sınıfın da bulunduğu kontrol edilmiştir.

---

## 4. Deneyler: Manuel ve Otomatik Hiperparametre Araması

### 4.1 Manuel arama (M1–M6)

Manuel adaylar, H1–H4 hipotezlerini tek-faktör kontrollü biçimde test edecek şekilde, nested CV başlamadan önce `configs/manual_configs.json` dosyasına kaydedilmiştir. Ortak varsayılanlar: batch size = 64, dropout = 0, aktivasyon = ReLU, weight decay = 1e-4, erken durdurma aktif, kırpma yok.

**Tablo 4a.** Manuel aday tanımları.

| Kod | Konfigürasyon | Amaç |
|---|---|---|
| M1 | AdamW, (32,), StandardScaler, LR = 0,001, WD = 1e-4, planlayıcı açık | Temel konfigürasyon |
| M2 | M1, optimizer = Adam | Optimizer etkisi |
| M3 | M1, optimizer = SGD + Momentum (0,9) | Optimizer etkisi |
| M4 | M1, scaler = MinMaxScaler | Ölçekleme etkisi |
| M5 | M1, gizli katmanlar = (64,32) | Mimari etkisi |
| M6 | M1, planlayıcı kapalı | LR düşürme etkisi |

**Tablo 4b.** Manuel adayların iç CV sonuçları (dış katlar ve tam development). Sütunlar: mean AP ± SD, süre, en iyi epoch. [SONUÇ — `results/manual_search_results.csv`]

### 4.2 Otomatik arama (R1–R6)

`ParameterSampler` (random_state = 31045) ile her optimizer için tam iki aday olmak üzere toplam altı aday örneklenmiş ve nested CV'den önce `configs/random_configs.json` dosyasına kaydedilmiştir. Arama uzayı Tablo 5a'da verilmiştir. SGD'nin aynı öğrenme oranında yavaş yakınsama olasılığı nedeniyle SGD için öğrenme oranı aralığı bir büyüklük mertebesi yukarı kaydırılmıştır.

**Tablo 5a.** Random Search arama uzayı.

| Hiperparametre | Uzay |
|---|---|
| Optimizer | SGD + Momentum, Adam, AdamW (her biri için 2 aday) |
| Gizli katmanlar | (16,), (32,), (64,), (64,32), (128,64) |
| Aktivasyon | ReLU, tanh |
| Öğrenme oranı | Adam/AdamW: 1e-4–1e-2; SGD: 1e-3–1e-1 (log-uniform) |
| Weight decay | 0, 1e-5, 1e-4, 1e-3 |
| Batch size | 32, 64, 128 |
| Ölçekleyici | Standard, MinMax, Robust |
| Dropout | 0,0, 0,2 |
| Aykırı değer kırpma | Yok, eğitim katında öğrenilmiş (1.–99. yüzdelik) |

**Tablo 5b.** Örneklenen R1–R6 adayları (tek `RandomState(31045)` akışı, sırayla SGD → Adam → AdamW; planlayıcı tüm adaylarda açık). Adaylar `configs/random_configs.json` dosyasına yazılmış ve nested CV'den önce commit edilmiştir; örneklenen hiçbir aday sonradan değiştirilmemiştir.

| Kod | Optimizer | Gizli katmanlar | Aktivasyon | LR | WD | Batch | Scaler | Dropout | Kırpma |
|---|---|---|---|---|---|---|---|---|---|
| R1 | SGD+M | (16,) | ReLU | 0,02737 | 1e-5 | 64 | MinMax | 0,2 | Yok |
| R2 | SGD+M | (32,) | ReLU | 0,00575 | 1e-5 | 128 | MinMax | 0,0 | Var |
| R3 | Adam | (64,) | tanh | 0,000459 | 1e-4 | 64 | MinMax | 0,0 | Yok |
| R4 | Adam | (16,) | ReLU | 0,0007506 | 1e-3 | 32 | Robust | 0,2 | Var |
| R5 | AdamW | (64,) | ReLU | 0,001886 | 1e-5 | 128 | Robust | 0,2 | Var |
| R6 | AdamW | (64,) | tanh | 0,0002112 | 1e-5 | 64 | Standard | 0,2 | Var |

**Tablo 5c.** R1–R6 iç CV sonuçları: mean AP ± SD, süre, en iyi epoch. [SONUÇ — `results/random_search_results.csv`]

**Şekil 4.** Örnek bir eğitimin eğitim ve early-stop doğrulama kayıpları ile epoch bazlı öğrenme oranı. [SONUÇ]

**Şekil 5.** Manuel ve otomatik adayların iç CV AP karşılaştırması. [SONUÇ]

---

## 5. Nested CV Bulguları

**Tablo 6.** Dış kat bazlı sonuçlar: manuel arama, otomatik arama, LogisticRegression, Dummy. Her dış katta seçilen aday da gösterilir. [SONUÇ — `results/outer_cv_results.csv`]

**Tablo 7.** Dış CV Mean ± SD (n = 3): AP, ROC-AUC, F1, Sensitivity, Specificity, Balanced Accuracy. [SONUÇ]

[Yorum: arama prosedürlerinin genelleme tahmini, referanslara göre kazanç, dış katlar arasında seçilen adayların tutarlılığı, H1–H4'ün desteklenip desteklenmediği.]

---

## 6. Nihai Seçim ve Bağımsız Test

### 6.1 Nihai konfigürasyon

[SONUÇ — tam development iç CV sonuçları, seçilen konfigürasyon, `results/final_hyperparameters.json`. Kilitli tahmin commit'inin hash'i.]

### 6.2 Test sonuçları

**Tablo 8.** Bağımsız test metrikleri (MLP ana seed, LogisticRegression) ve %95 bootstrap güven aralıkları. [SONUÇ — `results/test_metrics.csv`]

**Şekil 6.** Bağımsız test confusion matrix'leri: MLP ve LogisticRegression. [SONUÇ]

**Şekil 7.** ROC ve Precision–Recall eğrileri: MLP ve LogisticRegression. [SONUÇ]

### 6.3 Seed kararlılığı

**Tablo 9.** Üç başlangıç seed'inin (31046, 31146, 31246) test performansı ve Mean ± SD. Ana sonuç önceden ilan edilen seed 31046'dır. [SONUÇ — `results/seed_stability_results.csv`]

### 6.4 Hata analizi

[SONUÇ — False Positive ve False Negative sayıları, bu örneklerin özellik profilleri (ör. `poutcome`, `contact`, `month`, makroekonomik dönem), olası nedenler.]

---

## 7. Tartışma

[SONUÇ — aşağıdaki başlıklar sonuçlarla doldurulacak.]

- **Hiperparametre etkisi:** Hangi faktörün AP'yi en çok etkilediği; H1–H4.
- **Optimizer davranışları:** Yakınsama hızı, en iyi epoch dağılımı, SGD'nin LR duyarlılığı, eğitim süreleri.
- **Manuel ve otomatik arama:** Eşit bütçede hangisi daha iyi genelledi; seçim tutarlılığı.
- **Genelleme:** Dış CV, development iç CV ve bağımsız test sonuçları arasındaki farklar ve olası nedenleri.
- **Aşırı öğrenme ve gereksiz karmaşıklık:** Büyük mimarilerin kazancı, MLP'nin LogisticRegression'a üstünlüğü olup olmadığı.
- **Sınırlamalar:** n = 3 kat nedeniyle sınırlı belirsizlik temsili; küçük test kümesi (90 pozitif örnek); sınıf dengesizliği; rastgele bölünme ve müşteri kimliği yokluğu nedeniyle kişi-bağımsız ve zamansal genellemenin kanıtlanamaması; `duration` sonrası oluşan kopyalar; test etiketleri kamuya açık olduğundan kilitli commit'in kör test garantisi değil, zaman sıralı bir kayıt olması.

---

## 8. Sonuç

[SONUÇ — kanıta dayalı 3–5 bulgu ve sınırlamalar.]

---

## Yapay Zekâ Kullanım Beyanı

Bu projede kod yapısının planlanması, bazı yardımcı fonksiyonların yazılması ve rapor taslağının hazırlanmasında bir yapay zekâ asistanından (Anthropic Claude) yararlanılmıştır. Tüm deneyler yazar tarafından çalıştırılmış; rapordaki tüm sayısal sonuçlar depodaki kodla üretilmiş ve yazar tarafından kontrol edilmiştir.

---

## Kaynakça

1. Moro, S., Rita, P. ve Cortez, P. (2014). *Bank Marketing* [Veri seti]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
2. Moro, S., Cortez, P. ve Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
3. Pedregosa, F. vd. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
5. scikit-learn. Cross-validation: evaluating estimator performance; Metrics and scoring. https://scikit-learn.org/stable/
6. Paszke, A. vd. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
7. PyTorch. ReduceLROnPlateau. https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
8. Bergstra, J. ve Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
9. Varma, S. ve Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
10. Saito, T. ve Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
11. Loshchilov, I. ve Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
12. Kingma, D. P. ve Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.

---

## Ek A. Ortam ve Tekrarlanabilirlik

- İşletim sistemi: Windows 11 Pro for Workstations (10.0.26200); İşlemci: Intel64 Family 6 Model 151; GPU: kullanılmadı (CPU).
- Python 3.13.1; torch 2.15.0; scikit-learn 1.9.1; pandas 3.1.1; numpy 2.5.4; matplotlib 3.11.2; seaborn 0.13.2.
- Çalıştırma sırası: `python -m src.data` → `01_eda.ipynb` → `02_nested_cv.ipynb` → `03_final_test.ipynb`.
