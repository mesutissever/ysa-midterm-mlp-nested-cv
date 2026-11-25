# Banka Pazarlama Verisinde MLP ile İkili Sınıflandırma: Kat-İçi Ön İşleme, Manuel ve Otomatik Hiperparametre Araması ve İç İçe Çapraz Doğrulama

**Ders:** Yapay Sinir Ağları (Yüksek Lisans) — Ara Sınav Projesi
**Öğrenci:** Mesut İşsever, 264329015
**GitHub:** https://github.com/mesutissever/ysa-midterm-mlp-nested-cv
**Tarih:** 30 Kasım 2026

---

## Özet

Bu çalışmada UCI Bank Marketing veri setinin `bank-additional.csv` sürümünde (4.119 gözlem, pozitif sınıf oranı %11), görüşme öncesinde bilinen 19 özellikle vadeli mevduat aboneliğini tahmin eden PyTorch MLP modelleri incelenmiştir. Veri %80 development / %20 bağımsız test olarak tabakalı biçimde bölünmüş; imputation, ölçekleme, aykırı değer kırpma ve one-hot kodlama yalnızca ilgili eğitim alt kümesinde öğrenilmiştir. Hipotez temelli altı manuel (M1–M6) ve Random Search ile örneklenen altı otomatik (R1–R6) konfigürasyon, aynı 3×3 iç içe çapraz doğrulama düzeninde Average Precision (AP) birincil metriğiyle karşılaştırılmıştır. Dış CV'de manuel arama AP = 0,429 ± 0,055, otomatik arama 0,418 ± 0,039, LogisticRegression 0,420 ± 0,028 ve Dummy 0,110 değerine ulaşmıştır. Tüm development iç CV'siyle seçilen M2 (Adam, 32 nöronlu tek gizli katman, 2.017 parametre) üç seed ile eğitilmiş ve test tahminleri etiketsiz olarak ayrı bir commit'te kilitlenmiştir. Bağımsız testte MLP AP = 0,406 (%95 GA: 0,313–0,519) ve ROC-AUC = 0,744 (0,682–0,807) elde etmiş; LogisticRegression ile AP farkı (+0,016; %95 GA: −0,014 ile +0,041) sıfırdan ayırt edilememiştir. Bulgular, bu özellik setinde MLP'nin doğrusal modele belirgin üstünlük sağlamadığını, nested CV tahmininin test sonucuyla tutarlı olduğunu ve 0,50 eşiğinde duyarlılığın düşük (0,22) kaldığını göstermektedir.

**Araştırma sorusu:** Görüşme öncesi bilinen müşteri ve makroekonomik özelliklerle vadeli mevduat aboneliğini tahmin eden bir MLP'nin performansı; ön işleme (ölçekleyici, aykırı değer kırpma), mimari, optimizer ve öğrenme oranı planlaması tercihlerine ne kadar duyarlıdır ve hipotez temelli manuel arama ile eşit bütçeli rastgele arama, sızıntısız iç içe çapraz doğrulama altında nasıl karşılaştırılır?

---

## 1. Giriş

Çok Katmanlı Algılayıcı (Multi-Layer Perceptron, MLP), doğrusal dönüşümlerin doğrusal olmayan aktivasyon fonksiyonlarıyla art arda uygulandığı ileri beslemeli bir yapay sinir ağıdır. Bu esneklik, modelin başarısını hiperparametre seçimlerine (mimari, optimizer, öğrenme oranı, düzenlileştirme, yığın boyutu) ve girdi ön işlemesine bağımlı kılar.

Hiperparametre optimizasyonu iki yolla yapılabilir: hipotezlere dayalı **manuel arama** ve arama uzayından aday örnekleyen **otomatik arama** (ör. Random Search). Adil karşılaştırma aynı bölünmeleri, metriği ve aday bütçesini gerektirir. Hiperparametre seçimi ile performans tahmini aynı veride yapıldığında skor iyimser biçimde yanlı olur; **iç içe çapraz doğrulama (nested cross-validation)** seçimi iç döngüde, performans tahminini dış döngüde yaparak bu yanlılığı azaltır.

Kullanılan Bank Marketing verisinde pozitif sınıf oranı yaklaşık %11'dir. Böyle **dengesiz** bir problemde doğruluk (accuracy) yanıltıcıdır: tüm örnekleri negatif tahmin eden bir model bile yaklaşık %89 doğruluğa ulaşır. Bu nedenle birincil metrik olarak pozitif sınıfa odaklanan **Average Precision (AP)** seçilmiş; ROC-AUC, F1, duyarlılık, özgüllük ve dengelenmiş doğruluk tamamlayıcı metrikler olarak raporlanmıştır.

Çalışmanın amaçları şunlardır: (i) veri sızıntısını önleyen, tekrarlanabilir bir deney akışı kurmak; (ii) altı hipotez temelli manuel ve altı rastgele aday konfigürasyonu aynı 3×3 nested CV düzeninde karşılaştırmak; (iii) seçilen modelin genelleme başarısını yalnızca bir kez kullanılan bağımsız test kümesinde, güven aralıkları ve seed kararlılığıyla birlikte ölçmek.

---

## 2. Veri Seti ve Keşifsel Veri Analizi

### 2.1 Veri seti ve kullanım kuralları

UCI Machine Learning Repository'deki Bank Marketing veri setinin yalnızca `bank-additional.csv` dosyası kullanılmıştır (Moro vd., 2014). Dosya 4.119 gözlem, 20 girdi özelliği ve ikili hedef değişken `y` içerir. Daha büyük `bank-additional-full.csv` sürümü kullanılmamıştır. Hedef değişken "yes" = 1 (pozitif), "no" = 0 (negatif) olarak kodlanmıştır.

Modelleme öncesinde, performans incelenmeden önce aşağıdaki kurallar sabitlenmiştir:

- **`duration` çıkarıldı.** Görüşme süresi görüşmeden önce bilinemez ve hedefle doğrudan ilişkilidir; tüm modellerden çıkarılmıştır. Böylece 19 girdi özelliği kalmıştır: 9 sayısal, 10 kategorik.
- **"unknown" değerleri NaN kabul edilmedi.** Kategorik değişkenlerdeki "unknown" kodu ayrı bir kategori olarak korunmuştur; bilginin yokluğu da bilgi taşıyabilir.
- **`pdays = 999` korundu.** Bu değer "müşteriyle daha önce temas kurulmamış" anlamına gelen özel bir koddur; sıradan aykırı değer olarak silinmemiştir.

### 2.2 Seed politikası ve veri bölünmesi

Temel seed, öğrenci numarasının son beş hanesine (29015) 2026 eklenerek **31041** olarak belirlenmiştir. Alt seed'ler temel seed'e sabit ofset eklenerek deterministik biçimde türetilmiş ve deneylerden önce `configs/seeds.json` dosyasında kayıt altına alınmıştır (Ek B, Tablo 0): bölünme 31041, dış CV 31042, iç CV 31043, early-stop ayrımı 31044, Random Search 31045, PyTorch başlangıcı 31046 (nihai seed'ler 31046, 31146, 31246), bootstrap 31047.

Veri, hedef değişkene göre tabakalı (stratified) olarak %80 development ve %20 bağımsız test kümesine ayrılmıştır (Tablo 1a). Test kümesi EDA, ön işleme fit işlemleri, hiperparametre seçimi ve eşik belirlemede hiç kullanılmamıştır. Orijinal satır indeksleri `results/dataset_split.csv` dosyasına kaydedilmiştir.

**Tablo 1a.** Development/test bölünmesi ve sınıf dağılımı.

| Küme | n | Pozitif (y=1) | Pozitif oranı |
|---|---|---|---|
| Development | 3.295 | 361 | %10,96 |
| Bağımsız test | 824 | 90 | %10,92 |
| **Toplam** | **4.119** | **451** | **%10,95** |

**Kopya kayıtlar.** Ham veride birebir kopya satır yoktur. `duration` çıkarıldıktan sonra 19 satır başka bir satırla aynı hâle gelmiştir (20 kopya grubu; birinde özellikler aynı, etiketler farklıdır). Bu satırlar farklı görüşmeleri temsil ettiğinden silinmemiştir; 6 grup hem development hem test kümesinde örnek içerir ve bu durum sınırlama olarak not edilmiştir.

**Kişi-bağımsız genelleme sınırlaması.** Veride müşteri kimliği bulunmadığından aynı müşterinin görüşmeleri farklı kümelere düşebilir; bölünme de zamansal değil rastgeledir. Bu nedenle test performansı gelecekteki kampanyalara veya yeni müşterilere genellemeyi kanıtlamaz.

### 2.3 Tanımlayıcı istatistikler

Bu bölümdeki tüm tablo ve şekiller yalnızca development verisinden (n = 3.295) `01_eda.ipynb` ile üretilmiştir (kaynak: `results/eda_numeric_desc.csv`, `results/eda_outliers.csv`, `results/eda_unknown_counts.csv`). Kategorik frekans grafikleri Ek B'dedir (Şekil 2c).

**Tablo 1b.** Sayısal özellikler (development). IQR dışı: [Q1 − 1,5·IQR, Q3 + 1,5·IQR] aralığı dışında kalan gözlem sayısı.

| Özellik | Ortalama | Medyan | SD | IQR | Min–Maks | Çarpıklık | IQR dışı n (%) |
|---|---|---|---|---|---|---|---|
| age | 40,03 | 38,0 | 10,36 | 15,0 | 18–88 | 0,74 | 32 (%1,0) |
| campaign | 2,54 | 2,0 | 2,58 | 2,0 | 1–35 | 3,98 | 186 (%5,6) |
| pdays | 961,93 | 999 | 188,27 | 0 | 0–999 | −4,88 | 123 (%3,7)* |
| previous | 0,19 | 0 | 0,53 | 0 | 0–6 | 4,07 | 473 (%14,4)* |
| emp.var.rate | 0,09 | 1,1 | 1,56 | 3,2 | −3,4–1,4 | −0,74 | 0 |
| cons.price.idx | 93,58 | 93,80 | 0,58 | 0,92 | 92,20–94,77 | −0,22 | 0 |
| cons.conf.idx | −40,50 | −41,8 | 4,61 | 6,3 | −50,8–(−26,9) | 0,28 | 36 (%1,1) |
| euribor3m | 3,63 | 4,857 | 1,73 | 3,62 | 0,635–5,045 | −0,73 | 0 |
| nr.employed | 5.166,7 | 5.191,0 | 73,9 | 129,0 | 4.963,6–5.228,1 | −1,09 | 0 |

\*`pdays` ve `previous` için IQR sıfırdır; "aykırı" olarak işaretlenen değerler aslında daha önce temas kurulmuş müşterilerdir ve gerçek aykırı değer değildir.

**Tablo 1c.** Özel kodlu değerler (development). Diğer kategorik değişkenlerde "unknown" bulunmamaktadır; sayısal eksik değer yoktur.

| Değişken | Kod | n | Oran |
|---|---|---|---|
| default | unknown | 627 | %19,03 |
| education | unknown | 144 | %4,37 |
| housing | unknown | 87 | %2,64 |
| loan | unknown | 87 | %2,64 |
| job | unknown | 33 | %1,00 |
| marital | unknown | 8 | %0,24 |
| pdays | 999 (önceki temas yok) | 3.172 | %96,27 |
| poutcome | nonexistent | 2.822 | %85,64 |

**Şekil 2a.** Development sınıf dağılımı (2.934 negatif, %89,0; 361 pozitif, %11,0) ve sayısal özellikler arasındaki Pearson korelasyon matrisi. `figures/eda_class_dist.png`, `figures/eda_corr.png`

**Şekil 2b.** Sayısal özelliklerin histogramları ve boxplot'ları. `figures/eda_numeric_hist.png`, `figures/eda_numeric_box.png`

**Yorum.** `campaign` ve `previous` belirgin biçimde sağa çarpıktır (Tablo 1b, Şekil 2b; çarpıklık ≈ 4). `campaign` için IQR kuralı gözlemlerin %5,6'sını işaretlemektedir; bu durum H1'deki kırpma ve RobustScaler seçeneklerinin gerekçesidir. `pdays` değerlerinin %96'sı 999 özel kodudur (Tablo 1c), bu yüzden yüzdelik kırpma bu değişkene uygulanmamıştır. Makroekonomik değişkenler arasında çok yüksek korelasyonlar vardır (Şekil 2a): `emp.var.rate`–`euribor3m` r = 0,97, `euribor3m`–`nr.employed` r = 0,94. Betimsel hedef karşılaştırmasında pozitif sınıfın daha düşük `euribor3m` (ortalama 2,14 vs 3,81) ve `emp.var.rate` (−1,17 vs 0,25) dönemlerinde yoğunlaştığı, `poutcome = success` grubunda pozitif oranının %64, `nonexistent` grubunda %8,5 olduğu görülmektedir. En sık "unknown" `default` değişkenindedir (%19). **Hiçbir özellik EDA'ya dayanarak elenmemiştir;** hedefle ilişkiye dayalı bir eleme model seçimi sayılacağından yalnızca iç CV içinde yapılabilirdi.

### 2.4 Deney hipotezleri

EDA bulgularına dayanarak, dış CV sonuçları görülmeden önce şu hipotezler belirlenmiştir:

- **H1 (ölçekleme / aykırı değer):** Sağa çarpık sayısal özellikler nedeniyle, eğitim katında öğrenilmiş yüzdelik kırpma veya RobustScaler, StandardScaler'a kıyasla daha yüksek iç CV AP değeri sağlar. MinMaxScaler aykırı değerlere duyarlı olduğundan StandardScaler'dan daha düşük AP verir (M1 ve M4 karşılaştırması).
- **H2 (optimizer):** Aynı öğrenme oranında (0,001) SGD+Momentum, uyarlamalı yöntemlere (Adam, AdamW) göre daha yavaş yakınsar ve 150 epoch sınırında daha düşük AP'ye ulaşır (M1, M2, M3 karşılaştırması). Daha geniş öğrenme oranı aralığına sahip rastgele aramada bu fark azalır.
- **H3 (mimari):** Yaklaşık 3.300 örnekli küçük veri setinde daha büyük mimari (64,32), tek gizli katmanlı (32,) mimariye göre anlamlı bir AP kazancı sağlamaz (M1 ve M5 karşılaştırması).
- **H4 (öğrenme oranı planlaması):** ReduceLROnPlateau kapatıldığında (M6) erken durdurma daha erken tetiklenir ve AP bir miktar düşer.

---

## 3. Yöntem

### 3.1 Deneysel akış ve veri sızıntısının önlenmesi

Deney akışı Şekil 1'de özetlenmiştir. Bağımsız test kümesi ayrıldıktan sonra tüm geliştirme kararları yalnızca development verisi üzerinde, iç içe çapraz doğrulamayla verilmiştir.

**Şekil 1.** Veri bölünmesi ve nested CV akış şeması. `figures/flow_nested_cv.png`

1. Development verisi `StratifiedKFold(n_splits=3, shuffle=True, random_state=31042)` ile üç **dış kata**; her dış katın eğitim bölümü (outer-train) aynı yöntemle (random_state=31043) üç **iç kata** ayrılır.
2. Her iç katın eğitim bölümü ikiye bölünür: **%85 "fit" alt kümesi** ön işleme parametrelerini öğrenir ve MLP ağırlıklarını günceller; **%15 tabakalı "early-stop monitor" alt kümesi** yalnızca erken durdurmayı ve öğrenme oranı planlayıcısını yönetir.
3. Her adayın AP değeri **iç doğrulama katında** hesaplanır. Manuel (M1–M6) ve otomatik (R1–R6) aramalar aynı iç katlarda **ayrı ayrı** yürütülür; her aramanın en yüksek ortalama AP'li adayı outer-train'de yeniden eğitilir ve dış değerlendirme katında **yalnızca bir kez** ölçülür.
4. Dış kat skorları arama prosedürlerinin genelleme performansını tahmin eder; konfigürasyonları değiştirmek veya stratejiler arasında seçim yapmak için **kullanılmaz**.
5. Nihai seçim için tam development verisinde iç CV yeniden çalıştırılır ve ortalama AP'si en yüksek konfigürasyon seçilir. Nihai model üç seed ile eğitilir; test tahminleri etiketsiz olarak ayrı bir commit ile kilitlenir ve test metrikleri yalnızca bundan sonra hesaplanır.

Eğitim bütçesi: 3 × 3 × 12 = 108 iç CV eğitimi, 6 dış kat yeniden eğitimi, tam development'ta 3 × 12 = 36 seçim eğitimi ve 3 nihai eğitim; toplam 153 MLP eğitimi (Ek B, Tablo 2).

### 3.2 Kat-içi ön işleme

Ön işleme, scikit-learn `Pipeline` ve `ColumnTransformer` ile tanımlanmıştır. Tüm `fit` işlemleri yalnızca ilgili MLP ağırlık eğitim alt kümesinde ("fit" alt kümesi) yapılır; early-stop alt kümesine, iç doğrulama katına, dış değerlendirme katına ve teste yalnızca `transform` uygulanır. Öğrenilen parametreler şunlardır: medyan değerleri, kategori sözlüğü, ölçekleyici istatistikleri ve kırpma eşikleri.

- **Sayısal özellikler:** medyan imputation (development verisinde sayısal eksik değer bulunmamaktadır; adım, pipeline'ın genelliği için korunmuştur) → isteğe bağlı yüzdelik kırpma (eşikler fit alt kümesinin 1. ve 99. yüzdeliklerinden öğrenilir) → ölçekleyici.
- **`pdays` kırpılmaz.** Değerlerin %96'sı 999 özel kodudur; yüzdelik kırpma yalnızca gerçek gün değerlerini bozardı. Karar hedefe bakılmadan, kodlama yapısına dayanılarak verilmiştir; kırpma açıkken diğer sekiz sayısal özelliğe uygulanır.
- **"unknown" ayrı kategori olarak sabitlenmiştir.** Karar hedefle ilişkiye bakılmadan önceden verilmiştir; kat-içi en sık sınıfla doldurma alternatifi bütçe nedeniyle değerlendirilmemiş ve sınırlama olarak not edilmiştir. Kategorik adımdaki en sık sınıf imputation'ı yalnızca gerçek NaN değerler içindir.
- **Ölçekleyiciler:** **Standardization** (`StandardScaler`) her özelliği fit alt kümesinin ortalamasını çıkarıp standart sapmasına bölerek z-skoruna dönüştürür, değer aralığını sınırlamaz. **Normalization** (`MinMaxScaler`) değerleri fit alt kümesindeki min–maks aralığına göre [0, 1] aralığına taşır ve aykırı değerlere duyarlıdır. `RobustScaler` medyanı çıkarıp IQR'ye böler; aykırı değerlere karşı daha dayanıklıdır.
- **Kategorik özellikler:** `OneHotEncoder(handle_unknown="ignore")` ile kodlanır; fit alt kümesinde görülmemiş bir kategori diğer kümelerde sıfır vektörüyle temsil edilir.

One-hot kodlamadan sonra girdi boyutu, fit alt kümesinde gözlenen kategori sayısına bağlıdır. Tüm development verisi için bu boyut 9 sayısal + 52 one-hot = **61**'dir; nadir kategoriler (ör. `education = illiterate`) bir alt kümede hiç bulunmazsa boyut 60'a düşebilir.

**Sızıntı kontrolü.** `src/preprocessing.py` içindeki otomatik kontrol, (i) ölçekleyici istatistiklerinin yalnızca fit alt kümesinin istatistiklerine eşit olduğunu, (ii) kırpma eşiklerinin fit alt kümesinin yüzdeliklerinden öğrenildiğini ve `pdays`'in kırpılmadığını, (iii) fit alt kümesinde görülmemiş bir kategorinin hata vermeden sıfır vektörle temsil edildiğini doğrulamaktadır.

### 3.3 MLP mimarisi ve eğitim

MLP'ler PyTorch ile kurulmuştur. Gizli katmanlar `Linear → aktivasyon (ReLU veya tanh) → Dropout` bloklarından oluşur. Çıkış katmanı tek bir logit üretir. Kayıp fonksiyonu `BCEWithLogitsLoss`'tur; olasılık tahmini için logit'e sigmoid uygulanır. Varsayılan sınıflandırma eşiği 0,50'dir.

**Öğrenilebilir parametre sayısı.** Girdi boyutu *d* ve gizli katman genişlikleri *h₁, h₂, …* olmak üzere parametre sayısı Σ (önceki katman × sonraki katman + sonraki katman) formülüyle hesaplanır. *d* = 61 için:
- (32,) mimarisi (M1): 61·32 + 32 + 32·1 + 1 = **2.017** parametre.
- (64,32) mimarisi (M5): 61·64 + 64 + 64·32 + 32 + 32·1 + 1 = **6.081** parametre.

Kesin değerler her eğitimde koddan hesaplanıp kaydedilmektedir; nihai modelin katman bazlı hesabı Şekil 3'te gösterilmiştir. Eğitim ayarlarının özeti Ek B'de (Tablo 3) verilmiştir.

**Şekil 3.** Nihai MLP (M2) mimari şeması ve katman bazında öğrenilebilir parametre hesabı (d = 61; 1.984 + 33 = 2.017). Şekil eğitilmiş modelin katmanlarından kodla üretilmiş, toplam modelin kayıtlı değeriyle doğrulanmıştır. `figures/mlp_architecture.png`

**Optimizer'lar.** SGD + Momentum (0,9) tek bir global öğrenme oranı kullanır ve öğrenme oranına en duyarlı yöntemdir. Adam her parametre için uyarlamalı adım büyüklüğü hesaplar; weight decay gradyana L2 cezası olarak eklenir. AdamW weight decay'i gradyandan ayrık biçimde doğrudan ağırlıklara uygular (Loshchilov ve Hutter, 2019). M1 (AdamW) ile M2 (Adam) karşılaştırması bu farkı ölçer.

**Diğer ayrıntılar.** Ağırlıklar PyTorch varsayılanıyla (Kaiming-uniform) ilklendirilmiştir; erken durdurmada `min_delta = 0`'dır. Sınıf ağırlıklandırması tek-faktör karşılaştırmayı bozacağından uygulanmamıştır. Fit/early-stop ayrımı tüm adaylarda aynı seed (31044) ile yapılır. Tüm eğitimler CPU'da yapılmıştır. Early stopping tetiklenmeden 150 epoch'a ulaşan eğitimler "yakınsama uyarısı", sonlu olmayan kayıp üretenler "başarısız" olarak kaydedilir ve sonuçlardan çıkarılmaz. Her eğitimden önce Python, NumPy ve PyTorch seed'leri ayarlanır ve deterministik algoritmalar etkinleştirilir.

Planlayıcı her epoch sonunda early-stop alt kümesindeki doğrulama kaybıyla güncellenir. Erken durdurma ve öğrenme oranı düşürme aynı alt kümeyi izler; iç CV doğrulama katı bu amaçla kullanılmaz. Erken durdurmada AP yerine BCE kaybı kullanılmıştır. BCE kaybı epoch'tan epoch'a daha düzgün değişir ve olasılık kalibrasyonunu da yansıtır. AP ise hiperparametre seçiminin birincil metriği olarak korunmuştur. Her eğitimde eğitim kaybı, iç doğrulama kaybı, epoch bazlı öğrenme oranı, en iyi epoch, süre ve uyarılar CSV olarak kaydedilir.

### 3.4 Referans modeller

- **DummyClassifier** (`strategy="prior"`): sınıf önceliğini tahmin eder. Dengesiz yapıya karşı alt sınır görevi görür.
- **LogisticRegression**: sabit ön işleme ile (StandardScaler + OneHotEncoder) doğrusal referans modeldir.

Referans modeller dış katlarda yalnızca outer-train üzerinde eğitilip dış değerlendirme katında ölçülmüş, ayrıca bağımsız testte nihai MLP ile karşılaştırılmıştır. Hiperparametre arama bütçesine dahil değildirler.

### 3.5 Değerlendirme metrikleri

Birincil metrik **AP**'dir (`average_precision_score`). AP, precision–recall eğrisinin adım fonksiyonu toplamıdır ve trapez yöntemiyle hesaplanan PR-AUC ile aynı isim altında sunulmamıştır. Tamamlayıcı metrikler ROC-AUC, Accuracy, Precision, Sensitivity (Recall), Specificity, F1, Balanced Accuracy ve eğitim süresidir. Dış CV sonuçları üç kat üzerinden "ortalama ± örnek standart sapma" (Mean ± SD, n = 3) olarak verilmiştir. Katlar bağımsız tekrarlar olmadığından SD bir güven aralığı gibi yorumlanmamalıdır. Bağımsız test için ROC-AUC, AP ve F1 metriklerine 1.000 tekrarlı yüzdelik bootstrap ile %95 güven aralığı hesaplanmıştır; her yeniden örneklemede iki sınıfın da bulunduğu kontrol edilmiştir.

---

## 4. Deneyler: Manuel ve Otomatik Hiperparametre Araması

### 4.1 Manuel arama (M1–M6)

Manuel adaylar (Tablo 4a), H1–H4 hipotezlerini tek-faktör kontrollü biçimde test edecek şekilde, nested CV başlamadan önce `configs/manual_configs.json` dosyasına kaydedilmiştir. Ortak varsayılanlar: batch size = 64, dropout = 0, aktivasyon = ReLU, weight decay = 1e-4, erken durdurma aktif, kırpma yok.

**Tablo 4a.** Manuel aday tanımları.

| Kod | Konfigürasyon | Amaç |
|---|---|---|
| M1 | AdamW, (32,), StandardScaler, LR = 0,001, WD = 1e-4, planlayıcı açık | Temel konfigürasyon |
| M2 | M1, optimizer = Adam | Optimizer etkisi |
| M3 | M1, optimizer = SGD + Momentum (0,9) | Optimizer etkisi |
| M4 | M1, scaler = MinMaxScaler | Ölçekleme etkisi |
| M5 | M1, gizli katmanlar = (64,32) | Mimari etkisi |
| M6 | M1, planlayıcı kapalı | LR düşürme etkisi |

**Tablo 4b.** Manuel adayların iç CV mean AP değerleri (her dış katın outer-train'i içinde ve tüm development üzerinde, 3 iç kat) ile tüm development iç CV'sine ait SD, eğitim süresi ve epoch bilgileri. Kaynak: `results/manual_search_results.csv`.

| Kod | Dış kat 1 | Dış kat 2 | Dış kat 3 | Development mean AP ± SD | ROC-AUC | Süre (sn/eğitim) | En iyi / durulan epoch | Uyarı* |
|---|---|---|---|---|---|---|---|---|
| M1 | 0,4625 | 0,3670 | 0,4120 | 0,4394 ± 0,0456 | 0,7751 | 1,06 | 19,7 / 34,7 | 0 |
| **M2** | 0,4610 | 0,3683 | **0,4136** | **0,4408 ± 0,0456** | 0,7753 | 1,01 | 19,7 / 34,7 | 0 |
| M3 | 0,4476 | **0,3910** | 0,4022 | 0,4318 ± 0,0413 | 0,7788 | 2,95 | 128,0 / 136,7 | 7 |
| M4 | 0,4446 | 0,3664 | 0,4017 | 0,4312 ± 0,0370 | 0,7675 | 1,18 | 25,7 / 40,7 | 0 |
| M5 | **0,4676** | 0,3503 | 0,4089 | 0,4301 ± 0,0406 | 0,7660 | 1,15 | 12,7 / 27,7 | 0 |
| M6 | 0,4625 | 0,3670 | 0,4089 | 0,4394 ± 0,0456 | 0,7751 | 1,08 | 19,7 / 34,7 | 0 |

\*Uyarı: 12 iç eğitimden (3 dış kat × 3 + development × 3) kaçında early stopping tetiklenmeden 150 epoch'a ulaşıldığı. Kalın: ilgili kapsamda seçilen aday. Başarısız (NaN) eğitim olmamıştır.

### 4.2 Otomatik arama (R1–R6)

`ParameterSampler` (random_state = 31045) ile her optimizer için tam iki aday olmak üzere toplam altı aday örneklenmiş (Tablo 5b) ve nested CV'den önce `configs/random_configs.json` dosyasına kaydedilmiştir. Arama uzayı (Ek B, Tablo 5a): gizli katmanlar (16,), (32,), (64,), (64,32), (128,64); ReLU/tanh; LR Adam/AdamW için 1e-4–1e-2, SGD için 1e-3–1e-1 (log-uniform); weight decay 0, 1e-5, 1e-4, 1e-3; batch 32/64/128; Standard/MinMax/Robust; dropout 0/0,2; kırpma yok/var. SGD'nin aynı öğrenme oranında yavaş yakınsama olasılığı nedeniyle SGD için öğrenme oranı aralığı bir büyüklük mertebesi yukarı kaydırılmıştır.

**Tablo 5b.** Örneklenen R1–R6 adayları (tek `RandomState(31045)` akışı, sırayla SGD → Adam → AdamW; planlayıcı tüm adaylarda açık). Adaylar `configs/random_configs.json` dosyasına yazılmış ve nested CV'den önce commit edilmiştir; örneklenen hiçbir aday sonradan değiştirilmemiştir.

| Kod | Optimizer | Gizli katmanlar | Aktivasyon | LR | WD | Batch | Scaler | Dropout | Kırpma |
|---|---|---|---|---|---|---|---|---|---|
| R1 | SGD+M | (16,) | ReLU | 0,02737 | 1e-5 | 64 | MinMax | 0,2 | Yok |
| R2 | SGD+M | (32,) | ReLU | 0,00575 | 1e-5 | 128 | MinMax | 0,0 | Var |
| R3 | Adam | (64,) | tanh | 0,000459 | 1e-4 | 64 | MinMax | 0,0 | Yok |
| R4 | Adam | (16,) | ReLU | 0,0007506 | 1e-3 | 32 | Robust | 0,2 | Var |
| R5 | AdamW | (64,) | ReLU | 0,001886 | 1e-5 | 128 | Robust | 0,2 | Var |
| R6 | AdamW | (64,) | tanh | 0,0002112 | 1e-5 | 64 | Standard | 0,2 | Var |

**Tablo 5c.** Otomatik adayların iç CV sonuçları. Kaynak: `results/random_search_results.csv`.

| Kod | Dış kat 1 | Dış kat 2 | Dış kat 3 | Development mean AP ± SD | ROC-AUC | Süre (sn/eğitim) | En iyi / durulan epoch | Uyarı* |
|---|---|---|---|---|---|---|---|---|
| R1 | 0,4385 | 0,3548 | 0,3932 | 0,4264 ± 0,0558 | 0,7728 | 0,83 | 17,7 / 32,7 | 0 |
| R2 | 0,4298 | 0,3585 | **0,4149** | 0,4299 ± 0,0337 | 0,7761 | 1,47 | 107,0 / 118,3 | 5 |
| R3 | 0,4541 | 0,3627 | 0,4072 | 0,4228 ± 0,0334 | 0,7663 | 1,85 | 46,7 / 61,7 | 0 |
| **R4** | 0,4282 | 0,3541 | 0,3855 | **0,4353 ± 0,0497** | 0,7785 | 3,05 | 38,3 / 53,3 | 0 |
| R5 | 0,4178 | 0,3543 | 0,4050 | 0,4074 ± 0,0349 | 0,7742 | 0,67 | 17,7 / 32,7 | 0 |
| R6 | **0,4664** | **0,3771** | 0,4008 | 0,4288 ± 0,0402 | 0,7732 | 2,15 | 47,0 / 62,0 | 0 |

Tüm deney (150 MLP eğitimi ve referans modeller) CPU üzerinde toplam 172 saniye sürmüştür. Optimizer bazında, SGD adayları (M3, R2) en uzun eğitimleri ve uyarıların tamamını üretmiştir.

**Şekil 4.** Dış kat 1, iç kat 1 için M1, M3, M6 ve development'ta en iyi otomatik aday (R4) eğitim/early-stop doğrulama BCE kayıpları ve epoch bazlı öğrenme oranı. `figures/training_curves.png`

**Şekil 5.** Manuel ve otomatik adayların iç CV mean AP ± SD değerleri (üç dış kat ve tüm development). `figures/search_comparison.png`

### 4.3 Hipotezlerin iç CV sonuçlarına göre değerlendirilmesi

Aşağıdaki karşılaştırmalar tüm development iç CV değerlerine dayanmaktadır (Tablo 4b, Tablo 5c, Şekil 5). Adaylar arasındaki farkların büyük bölümü kat SD'lerinden (≈0,04–0,05) küçüktür; bu nedenle bulgular eğilim olarak yorumlanmalıdır.

- **H1 (ölçekleme/kırpma): kısmen desteklendi.** MinMaxScaler (M4: 0,4312) StandardScaler'ın (M1: 0,4394) gerisinde kalmıştır. Otomatik aramada RobustScaler + kırpma kullanan R4 en yüksek otomatik skoru (0,4353) verirken aynı kombinasyonu kullanan R5 en düşük skoru (0,4074) vermiştir; kırpma/robust ölçeklemenin tek başına belirleyici olmadığı, diğer hiperparametrelerle etkileştiği görülmektedir.
- **H2 (optimizer): yakınsama hızı açısından desteklendi, AP açısından zayıf.** LR = 0,001 ile SGD+Momentum (M3) ortalama 128 epoch'ta en iyi noktasına ulaşmış, 12 eğitimin 7'sinde 150 epoch sınırına dayanmış ve Adam/AdamW'ye göre yaklaşık 3 kat uzun sürmüştür; buna karşın AP kaybı küçüktür (0,4318). Daha büyük LR ile SGD (R1, LR = 0,027) 17,7 epoch'ta yakınsamıştır. Adam (M2) ile AdamW (M1) neredeyse aynı sonucu vermiştir; WD = 1e-4 düzeyinde ayrık ve bağlı weight decay arasındaki fark ihmal edilebilir kalmıştır.
- **H3 (mimari): desteklendi.** (64,32) mimarisi (M5: 0,4301, ≈6.060 parametre) tek katmanlı (32,) mimariye (M1: 0,4394, ≈2.006 parametre) göre kazanç sağlamamış, en iyi epoch'a daha erken (12,7) ulaşarak daha hızlı aşırı öğrenme eğilimi göstermiştir.
- **H4 (LR planlaması): desteklenmedi.** M6 (planlayıcı kapalı) 12 iç eğitimin 11'inde M1 ile birebir aynı AP'yi vermiştir. Bunun nedeni, ilk öğrenme oranı düşüşünün en iyi epoch'tan sonra gerçekleşmesi ve en iyi ağırlıklar geri yüklendiğinden planlayıcının nihai modele etki etmemesidir. Bu ayarlarda (planlayıcı patience = 5, early stopping patience = 15) ReduceLROnPlateau pratikte etkisiz kalmıştır.

---

## 5. Nested CV Bulguları

Kat bazlı ham sonuçlar Tablo 6'da, üç kat üzerinden özet Tablo 7'de verilmiştir. Dış kat 2'de seçilen M3'ün refit eğitimi 150 epoch sınırına ulaşmıştır (yakınsama uyarısı).

**Tablo 6.** Dış kat bazlı sonuçlar (eşik 0,50). Her dış katta iç CV ile seçilen aday outer-train'de yeniden eğitilip outer-eval'de bir kez ölçülmüştür. Kaynak: `results/outer_cv_results.csv`.

| Dış kat | Yöntem | Seçilen | İç CV AP | AP | ROC-AUC | F1 | Sensitivity | Specificity | TP / FP / FN |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Manuel arama | M5 | 0,4676 | 0,3666 | 0,7500 | 0,2674 | 0,1901 | 0,9714 | 23 / 28 / 98 |
| 1 | Otomatik arama | R6 | 0,4664 | 0,3739 | 0,7485 | 0,2874 | 0,1983 | 0,9775 | 24 / 22 / 97 |
| 1 | LogisticRegression | – | – | 0,3915 | 0,7587 | 0,3333 | 0,2397 | 0,9755 | 29 / 24 / 92 |
| 1 | Dummy | – | – | 0,1101 | 0,5000 | 0,0000 | 0,0000 | 1,0000 | 0 / 0 / 121 |
| 2 | Manuel arama | M3 | 0,3910 | 0,4554 | 0,7918 | 0,3185 | 0,2083 | 0,9877 | 25 / 12 / 95 |
| 2 | Otomatik arama | R6 | 0,3771 | 0,4500 | 0,7810 | 0,3855 | 0,2667 | 0,9857 | 32 / 14 / 88 |
| 2 | LogisticRegression | – | – | 0,4478 | 0,7745 | 0,4192 | 0,2917 | 0,9877 | 35 / 12 / 85 |
| 2 | Dummy | – | – | 0,1093 | 0,5000 | 0,0000 | 0,0000 | 1,0000 | 0 / 0 / 120 |
| 3 | Manuel arama | M2 | 0,4136 | 0,4659 | 0,7987 | 0,3478 | 0,2333 | 0,9867 | 28 / 13 / 92 |
| 3 | Otomatik arama | R2 | 0,4149 | 0,4290 | 0,7888 | 0,3333 | 0,2250 | 0,9847 | 27 / 15 / 93 |
| 3 | LogisticRegression | – | – | 0,4207 | 0,7890 | 0,3432 | 0,2417 | 0,9796 | 29 / 20 / 91 |
| 3 | Dummy | – | – | 0,1093 | 0,5000 | 0,0000 | 0,0000 | 1,0000 | 0 / 0 / 120 |

**Tablo 7.** Dış CV Mean ± SD (n = 3, örnek SD). Kaynak: `results/outer_cv_summary_formatted.csv`.

| Yöntem | AP | ROC-AUC | F1 | Sensitivity | Specificity | Balanced Acc. | Precision | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Manuel arama | 0,429 ± 0,055 | 0,780 ± 0,026 | 0,311 ± 0,041 | 0,211 ± 0,022 | 0,982 ± 0,009 | 0,596 ± 0,015 | 0,603 ± 0,132 | 0,897 ± 0,010 |
| Otomatik arama | 0,418 ± 0,039 | 0,773 ± 0,021 | 0,335 ± 0,049 | 0,230 ± 0,034 | 0,983 ± 0,004 | 0,606 ± 0,019 | 0,620 ± 0,089 | 0,900 ± 0,008 |
| LogisticRegression | 0,420 ± 0,028 | 0,774 ± 0,015 | 0,365 ± 0,047 | 0,258 ± 0,029 | 0,981 ± 0,006 | 0,619 ± 0,018 | 0,628 ± 0,104 | 0,902 ± 0,009 |
| Dummy (prior) | 0,110 ± 0,000 | 0,500 ± 0,000 | 0,000 ± 0,000 | 0,000 ± 0,000 | 1,000 ± 0,000 | 0,500 ± 0,000 | 0,000 ± 0,000 | 0,890 ± 0,000 |

**Yorum.** Her iki arama prosedürü Dummy alt sınırının (AP ≈ 0,11) yaklaşık dört katı AP'ye ulaşmıştır. Buna karşın manuel arama, otomatik arama ve LogisticRegression arasındaki AP farkları katlar arası SD'nin çok altındadır; MLP'nin doğrusal modele belirgin üstünlüğü gösterilememiştir. Eşik 0,50'de tüm modeller yüksek özgüllük (≈0,98) fakat düşük duyarlılık (≈0,21–0,26) üretir; dengesiz yapıda olasılıklar nadiren 0,50'yi aşar. Accuracy (≈0,90) Dummy'nin 0,890'ından neredeyse ayırt edilemez; bu, accuracy'nin neden birincil metrik olmadığını doğrular.

Dış katlarda seçilen adaylar tutarlı değildir (manuel: M5, M3, M2; otomatik: R6, R6, R2); adaylar arasındaki farklar küçük olduğundan seçim kat değişimine duyarlıdır. Dış kat 1'de iç CV AP (≈0,47) outer-eval AP'sinden (≈0,37) belirgin biçimde yüksektir: seçilen adayın iç CV skoru seçim yanlılığı nedeniyle iyimserdir (kat bazlı grafik: Ek B, Şekil 5b). Dış kat sonuçları hiçbir konfigürasyonun değiştirilmesinde veya nihai stratejinin seçiminde kullanılmamıştır.

### 5.1 Tüm development üzerinde nihai seçim

Önceden tanımlanan kural gereği (tüm development iç CV mean AP en yüksek aday), manuel aramada **M2** (0,4408 ± 0,0456), otomatik aramada **R4** (0,4353 ± 0,0497) en iyi olmuş; nihai konfigürasyon olarak **M2** seçilmiştir: Adam, (32,), ReLU, LR = 0,001, WD = 1e-4, batch 64, StandardScaler, dropout 0, kırpma yok, planlayıcı açık. M2 ile M1 arasındaki fark (0,0014) çok küçüktür; seçim, kuralın mekanik uygulamasıdır ve M2'nin M1'den üstün olduğu biçiminde yorumlanmamalıdır. Kaynak: `results/dev_selection.json`.

---

## 6. Nihai Seçim ve Bağımsız Test

### 6.1 Nihai konfigürasyon, eğitim ve kilitleme

Seçilen M2 konfigürasyonu tüm development verisinde eğitilmiştir: 2.800 örneklik fit alt kümesi ve 495 örneklik tabakalı early-stop alt kümesi (seed 31044); ön işleme yalnızca fit alt kümesinde öğrenilmiştir. Konfigürasyon önceden belirlenen üç ağırlık başlangıç seed'iyle eğitilmiş; ana seed 31046 kilitlemeden önce ilan edilmiştir.

Üç eğitim de (Ek B, Tablo 8a) en iyi epoch 18–20'de, early stopping ile durmuştur; yakınsama uyarısı veya başarısız eğitim yoktur. Öğrenme oranı iki kez yarıya inmiş, ancak her iki düşüş de en iyi epoch'tan sonra gerçekleşmiştir (Ek B, Şekil 4b). Bu, H4 bulgusuyla tutarlıdır. Her seed için model ağırlıkları, fit edilmiş ön işleme nesnesi ve öğrenilmiş dönüşüm istatistikleri (medyanlar, ölçekleyici ortalama/SD değerleri, kategori sözlüğü) test etiketleri okunmadan `models/` klasörüne kaydedilmiştir. Nihai hiperparametreler, eşik (0,50), seed'ler ve kitaplık sürümleri `results/final_hyperparameters.json` dosyasındadır.

**Kilitleme kaydı.** Nihai kod `8a81da2` (17.11.2026 19:46) commit'inde sabitlenmiş; etiketsiz `test_predictions_locked.csv` (824 satır; üç MLP seed'i ve LogisticRegression için olasılık ve sınıf tahmini), sha256 manifest'i (`0acd5903…b451d1`) ve modeller `cb19a1c` (18.11.2026 10:57) commit'iyle yüklenmiştir. Test metrikleri ancak bundan sonra `50b5b02` commit'inde hesaplanmıştır (Ek B, Tablo 8c).

Değerlendirme kodu, kilitli dosya git'te commit edilmemişse, commit'ten sonra değiştirilmişse veya sha256 manifest ile uyuşmuyorsa test etiketlerini okumayı reddeder. Test etiketleri kamuya açık olduğundan bu işlem kör test garantisi değil, zaman sıralı bir deney kaydıdır.

### 6.2 Test sonuçları

**Tablo 8.** Bağımsız test metrikleri (n = 824, 90 pozitif; eşik 0,50). Köşeli parantez içinde 1.000 tekrarlı yüzdelik bootstrap %95 güven aralığı (seed 31047; tek sınıflı yeniden örnekleme oluşmamıştır). Kaynak: `results/test_metrics.csv`, `results/test_bootstrap_ci.csv`.

| Metrik | MLP (M2, seed 31046) | LogisticRegression | Dummy (prior) |
|---|---|---|---|
| **AP** | **0,406** [0,313; 0,519] | 0,390 [0,300; 0,509] | 0,109 |
| ROC-AUC | 0,744 [0,682; 0,807] | 0,737 [0,675; 0,802] | 0,500 |
| F1 | 0,320 [0,215; 0,431] | 0,333 [0,222; 0,439] | 0,000 |
| Accuracy | 0,897 | 0,898 | 0,891 |
| Precision | 0,571 | 0,583 | 0,000 |
| Sensitivity | 0,222 | 0,233 | 0,000 |
| Specificity | 0,980 | 0,980 | 1,000 |
| Balanced Accuracy | 0,601 | 0,606 | 0,500 |
| TP / FP / TN / FN | 20 / 15 / 719 / 70 | 21 / 15 / 719 / 69 | 0 / 0 / 734 / 90 |

Tablo 8'e göre MLP sıralama metriklerinde (AP, ROC-AUC) LogisticRegression'dan biraz yüksek, eşik tabanlı F1'de biraz düşüktür. Eşleştirilmiş bootstrap farkları (MLP − LogisticRegression; Ek B, Tablo 8b): AP +0,016 [−0,014; +0,041], ROC-AUC +0,007 [−0,009; +0,022], F1 −0,013 [−0,062; +0,031]. Üç aralık da sıfırı içerir; bu test kümesinde iki model **istatistiksel olarak ayırt edilememektedir**. İki modelin 0,50 eşiğindeki sınıf tahminleri 824 örneğin yalnızca 5'inde farklıdır. Her iki model Dummy alt sınırının (AP = 0,109) yaklaşık 3,7 katı AP'ye ulaşmıştır; accuracy (0,897) ise Dummy'nin 0,891 değerinden neredeyse ayırt edilemez.

**Şekil 6.** Bağımsız test karışıklık matrisleri (eşik 0,50): MLP ve LogisticRegression. `figures/test_confusion.png`

Şekil 6'da her iki modelin hata yapısı neredeyse aynıdır: negatiflerin %98'i doğru sınıflandırılırken 90 pozitifin yalnızca 20–21'i yakalanmıştır. Hataların büyük çoğunluğu yanlış negatiftir (70 FN'ye karşı 15 FP).

**Şekil 7.** Bağımsız test ROC ve Precision–Recall eğrileri: MLP ve LogisticRegression. PR grafiğindeki kesikli çizgi pozitif sınıf oranını (0,109) gösterir. `figures/test_roc_pr.png`

Şekil 7'de iki modelin eğrileri büyük ölçüde üst üste binmektedir. Düşük duyarlılık bölgesinde (recall < 0,2) kesinlik 0,6–0,8 düzeyindedir; recall 0,55–0,6'yı geçtikten sonra kesinlik hızla taban oranına iner. Pozitiflerin yaklaşık yarısı yüksek olasılıklı küçük bir grupta toplanmakta, geri kalanı negatiflerden ayırt edilememektedir.

### 6.3 Seed kararlılığı

**Tablo 9.** Nihai konfigürasyonun üç başlangıç seed'iyle test performansı. Ana sonuç önceden ilan edilen seed 31046'dır; diğer seed'ler yalnızca kararlılığı betimlemek için raporlanmıştır, seçimde kullanılmamıştır. Kaynak: `results/seed_stability_results.csv`, `results/seed_stability_summary.csv`.

| Seed | AP | ROC-AUC | F1 | Sensitivity | Specificity | Balanced Acc. | Precision |
|---|---|---|---|---|---|---|---|
| 31046 (ana) | 0,406 | 0,744 | 0,320 | 0,222 | 0,980 | 0,601 | 0,571 |
| 31146 | 0,405 | 0,747 | 0,309 | 0,211 | 0,981 | 0,596 | 0,576 |
| 31246 | 0,396 | 0,747 | 0,306 | 0,211 | 0,980 | 0,595 | 0,559 |
| **Mean ± SD** | **0,403 ± 0,006** | **0,746 ± 0,002** | **0,312 ± 0,007** | **0,215 ± 0,006** | **0,980 ± 0,001** | **0,597 ± 0,003** | **0,569 ± 0,009** |

Seed'e bağlı değişkenlik çok küçüktür (Tablo 9): AP için SD = 0,006, bootstrap SD'sinin (0,054) yaklaşık onda biridir. Bu nedenle test sonucundaki belirsizliğin baskın kaynağı ağırlık başlangıcı değil, küçük test kümesinin örnekleme değişkenliğidir. Ana seed üç seed arasında en yüksek AP'yi vermiştir, ancak fark (0,406'ya karşı ortalama 0,403) ihmal edilebilir düzeydedir.

### 6.4 Hata analizi

Ana modelde 15 yanlış pozitif (FP) ve 70 yanlış negatif (FN) vardır. Gerçek pozitiflerin %78'i kaçırılmıştır. Kaynak: `results/test_error_analysis.csv`, `results/test_fn_by_category.csv`, `results/test_calibration_bins.csv`.

- **FN'ler:** Model, 0,50 eşiğinde yalnızca önceki kampanyası başarılı (`poutcome = success`) müşterileri pozitif tahmin edebilmektedir: bu gruptaki 22 pozitifin 20'si yakalanmış, daha önce hiç aranmamış (51) veya önceki kampanyası başarısız (17) 68 pozitifin tamamı kaçırılmıştır (Ek B, Tablo 10). En emin FN'ler (olasılık ≈ 0,02–0,03) ortak bir profile sahiptir: önceki temas yok, 10'undan 8'ine `telephone` ile ulaşılmış, çoğunlukla Mayıs–Haziran ayları ve yüksek faiz dönemi (`euribor3m` ≈ 4,1–5,0). Görüşme öncesi özellikler bu grubun içindeki aboneleri ayırt edecek bilgi içermemektedir; bu bilgi büyük ölçüde modelden çıkarılan `duration` değişkenindedir.
- **FP'ler:** En yüksek olasılıklı 10 FP'nin 9'u yakın zamanda temas kurulmuş (`pdays` < 20) müşterilerdir; 7'sinin önceki kampanyası başarılıdır ve hepsi düşük faiz döneminde (`euribor3m` < 1,3) aranmıştır. Model bu güçlü sinyale dayanarak yüksek olasılık (0,63–0,80) vermiş, ancak bu müşteriler abone olmamıştır; hatalar modelin öğrendiği örüntüyle tutarlıdır.
- **Kalibrasyon (betimsel):** Testte ortalama tahmin olasılığı 0,106, gözlenen pozitif oranı 0,109'dur; en üst beşte birlik dilimde 0,339'a karşı 0,321. Düşük duyarlılığın nedeni kalibrasyon bozukluğu değil, pozitiflerin çoğuna gerçekten düşük olasılık düşmesi ve 0,50 eşiğinin bu dengesiz problemde yüksek olmasıdır.
- **Eşik:** Eşiği düşürmek duyarlılığı artırır, kesinliği azaltır. Eşik yalnızca development iç CV tahminleriyle seçilebilirdi; bu çalışmada eşik araması yapılmamış ve test sonuçlarına bakılarak eşik değiştirilmemiştir.

## 7. Tartışma

### 7.1 Hiperparametre etkisi ve optimizer davranışları

İç CV'de en iyi ve en kötü manuel aday arasındaki fark yalnızca 0,011 AP'dir (M2 0,441; M5 0,430). Bu fark katlar arası SD'nin (≈0,045) dörtte biridir. Otomatik adaylarda aralık daha geniştir (R5 0,407; R4 0,435), ancak yine SD mertebesindedir. Dolayısıyla bu veri setinde MLP performansı incelenen hiperparametrelere **düşük duyarlılık** göstermektedir. Sonuçları belirleyen asıl etken, mimari veya optimizer tercihinden çok mevcut özelliklerin taşıdığı sınırlı bilgidir.

En belirgin etki performansta değil **yakınsama davranışında** görülmüştür (Bölüm 4.3): LR = 0,001 ile SGD+Momentum (M3) Adam/AdamW'ye göre yaklaşık 6 kat fazla epoch ve 3 kat uzun süre gerektirmiş, AP kaybı küçük kalmıştır. Yüksek öğrenme oranlı R1'in hızla yakınsaması, SGD için arama uzayının yukarı kaydırılmasını doğrular. ReduceLROnPlateau en iyi epoch'tan sonra devreye girdiği için nihai modeli etkilememiştir (H4); etkili olması için planlayıcı patience değerinin daha küçük olması gerekirdi, ancak bu ayarlar ödevde sabittir.

### 7.2 Aşırı öğrenme ve gereksiz karmaşıklık

Daha büyük mimari (64,32) yaklaşık üç kat fazla parametreyle AP kazancı sağlamamış ve en iyi epoch'a daha erken ulaşmıştır (12,7'ye karşı 19,7). Eğitim eğrilerinde (Şekil 4; Ek B, Şekil 4b) eğitim kaybı düşmeye devam ederken early-stop kaybı yaklaşık 20. epoch'tan sonra yatay seyreder; erken durdurma ve en iyi ağırlıkların geri yüklenmesi aşırı öğrenmeyi sınırlamıştır. 2.017 parametreli tek katmanlı MLP bile dış CV'de ve bağımsız testte LogisticRegression'dan ayırt edilemeyen sonuçlar üretmiştir: sinyal büyük ölçüde birkaç güçlü ve doğrusala yakın özellikte (`poutcome`, `nr.employed`, `euribor3m`) toplanmıştır. Uygulamada yorumlanabilirlik ve maliyet açısından LogisticRegression tercih edilebilir.

### 7.3 Manuel ve otomatik arama

Eşit aday bütçesinde (6'ya 6) manuel arama dış CV'de biraz daha yüksek AP (0,429'a karşı 0,418), otomatik arama biraz daha yüksek F1 ve duyarlılık üretmiştir; farklar SD'lerin altındadır ve iki stratejiden birinin daha iyi genellediği söylenemez. Manuel aramanın avantajı, tek-faktör kontrollü adaylarla her hipotezin doğrudan test edilebilmesidir. Otomatik arama altı adayla geniş bir uzayı çok seyrek örneklemiştir; örneğin (128,64) mimarisi hiç örneklenmemiştir.

### 7.4 Genelleme: iç CV, dış CV ve bağımsız test

Üç düzey ayrı yorumlanmalıdır (Ek B, Tablo 11). **Development iç CV skoru** (M2: 0,441 ± 0,046) seçim için kullanılmıştır; 12 aday arasından en yükseğinin seçilmesi nedeniyle iyimser yanlıdır (Varma ve Simon, 2006). **Dış CV** (0,429 ± 0,055), seçim prosedürünün tamamının genelleme performansını tahmin eder; bu çalışmanın asıl performans tahminidir. **Bağımsız test** (0,406) ise tek bir nihai modelin, hiç kullanılmamış 824 örnekteki performansıdır.

Test AP'si development iç CV skorundan düşük, dış CV tahmininin ise içindedir. LogisticRegression için de benzer bir düşüş görülmektedir (dış CV 0,420; test 0,390). Bu, farkın MLP'ye özgü bir aşırı öğrenmeden değil, büyük olasılıkla bu test örnekleminin biraz daha zor olmasından kaynaklandığını düşündürür. Testteki AP güven aralığının genişliği (≈ ±0,10) dikkate alındığında fark örnekleme değişkenliğiyle açıklanabilir; bu tutarlılık, nested CV'nin genelleme performansını iyimser göstermediğini destekler.

### 7.5 Sınırlamalar

- **Sınırlı belirsizlik temsili:** Dış CV yalnızca 3 kattan oluşur; katlar bağımsız tekrarlar olmadığından Mean ± SD bir güven aralığı değildir.
- **Küçük test kümesi:** Testte yalnızca 90 pozitif vardır; bootstrap aralıkları geniştir (AP için ≈ 0,21) ve modeller arasındaki küçük farklar test edilememektedir.
- **Sınıf dengesizliği ve eşik:** Sınıf ağırlıklandırması ve eşik optimizasyonu yapılmadığından 0,50 eşiğinde duyarlılık düşüktür (≈0,22); eşik development iç CV tahminleriyle seçilseydi artırılabilirdi.
- **Kişi-bağımsız ve zamansal genelleme:** Müşteri kimliği yoktur ve bölünme rastgeledir; `duration` çıkarıldıktan sonra oluşan 19 kopya satırın bir kısmı development ile test arasında paylaşılır. Makroekonomik değişkenler dönemi kodladığından zamana göre bölünmüş bir değerlendirme daha düşük performans verebilir.
- **Arama kapsamı:** Otomatik arama yalnızca 6 adayla sınırlıdır. "unknown" için kat-içi imputation alternatifi, sınıf ağırlıklandırması ve eşik araması değerlendirilmemiştir.
- **Kör test değil:** Test etiketleri kamuya açıktır; kilitli commit yalnızca tahminlerin metriklerden önce sabitlendiğini gösteren zaman sıralı bir kayıttır.

## 8. Sonuç

Bu çalışmada Bank Marketing verisinde MLP sınıflandırıcıları için veri sızıntısını önleyen, tekrarlanabilir bir seçim ve değerlendirme süreci kurulmuştur. Kanıta dayalı başlıca bulgular şunlardır:

1. **Seçim süreci tutarlıdır.** Nested CV'nin manuel arama için verdiği dış CV tahmini (AP = 0,429 ± 0,055), seçilen modelin bağımsız test sonucuyla (0,406; %95 GA 0,313–0,519) tutarlıdır. Seçim skorunun (0,441) iyimser yanlılığı dış CV ile görünür hâle gelmiştir.
2. **MLP doğrusal referansı geçememiştir.** Hem dış CV'de hem bağımsız testte MLP ile LogisticRegression arasındaki farklar belirsizlik sınırları içindedir (test ΔAP = +0,016; %95 GA −0,014 ile +0,041). Her iki model de Dummy alt sınırının yaklaşık dört katı AP üretmektedir.
3. **Hiperparametre duyarlılığı düşüktür.** Manuel adaylar arasındaki AP farkları kat SD'sinin altındadır. Hipotezlerden H3 (büyük mimari kazanç sağlamaz) desteklenmiş, H1 kısmen desteklenmiş, H2 yalnızca yakınsama hızı açısından desteklenmiştir (SGD, LR = 0,001 ile yaklaşık 6 kat daha fazla epoch gerektirmiştir). H4 desteklenmemiştir: planlayıcı en iyi epoch'tan sonra devreye girdiği için etkisiz kalmıştır.
4. **Model kararlıdır, ancak eşik 0,50'de duyarlılığı düşüktür.** Seed'e bağlı test AP değişkenliği küçüktür (SD = 0,006). Olasılıklar makul ölçüde kalibredir, ancak pozitiflerin %78'i kaçırılmaktadır. Model ağırlıklı olarak önceki kampanya başarısı ve makroekonomik dönem sinyaline dayanmakta, ilk kez aranan müşteriler arasındaki aboneleri ayırt edememektedir.

Bu sonuçlar küçük test kümesi, üç katlı dış CV, rastgele (zamansal olmayan) bölünme ve müşteri kimliği yokluğu nedeniyle sınırlıdır. Gelecek çalışmada eşiğin ve sınıf ağırlıklandırmasının development iç CV içinde seçilmesi, zamana göre bölünmüş bir değerlendirme ve daha geniş bir otomatik arama bütçesi önerilmektedir.

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
- Çalıştırma sırası: `python -m src.data` → `01_eda.ipynb` → `02_nested_cv.ipynb` → `03_final_test.ipynb`. Ham veri bulunmazsa ilk okumada UCI'den indirilir ve sha256 ile doğrulanır.
- Toplam süre: nested CV (150 MLP eğitimi ve referans modeller) 172 saniye; nihai eğitimler 3 seed için birkaç saniye (CPU).
- Kilitleme commit'leri: kod `8a81da2`, kilitli tahminler `cb19a1c`, test metrikleri `50b5b02`.

---

## Ek B. Ek Tablolar ve Şekiller

Bu ekteki tablo ve şekiller ana metinde anılan ayrıntılardır; tamamı depodaki kodla üretilmiştir.

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

**Tablo 2.** Eğitim bütçesi.

| Hesap | Eğitim sayısı |
|---|---|
| 3 dış × 3 iç × (6 manuel + 6 otomatik) | 108 |
| Dış katlarda seçilen iki yöntemin yeniden eğitimi | 6 |
| Tam development üzerinde 3 iç × 12 aday | 36 |
| Nihai model, 3 seed | 3 |
| **Toplam MLP eğitimi** | **153** |

**Şekil 2c.** Kategorik özelliklerin frekans dağılımları ("unknown" dahil). `figures/eda_categorical_bars.png`

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

**Tablo 8a.** Nihai eğitimler. Kaynak: `results/final_training_summary.csv`.

| Seed | Parametre | En iyi / durulan epoch | En iyi early-stop BCE | Son LR |
|---|---|---|---|---|
| **31046 (ana)** | 2.017 | 20 / 35 | 0,2628 | 2,5e-4 |
| 31146 | 2.017 | 18 / 33 | 0,2605 | 2,5e-4 |
| 31246 | 2.017 | 18 / 33 | 0,2662 | 2,5e-4 |

**Şekil 4b.** Nihai modelin üç seed için eğitim ve early-stop doğrulama BCE kaybı ile epoch bazlı öğrenme oranı. `figures/final_training_curves.png`

**Tablo 8c.** Kilitleme commit'leri.

| Commit | Tarih | İçerik |
|---|---|---|
| `8a81da2` | 17.11.2026 19:46 | Nihai eğitim ve kilitli tahmin kodu |
| `cb19a1c` | 18.11.2026 10:57 | Etiketsiz `test_predictions_locked.csv` (824 satır; 3 MLP seed'i + LogisticRegression için olasılık ve sınıf tahmini), manifest (sha256 `0acd5903…b451d1`), modeller |
| `50b5b02` | 18.11.2026 11:28 | Test metrikleri (yalnızca kilit commit'i doğrulandıktan sonra) |

**Şekil 5b.** Dış CV AP: kat bazlı değerler ve ortalama ± SD. `figures/outer_cv_ap.png`

**Tablo 8b.** Eşleştirilmiş bootstrap farkı (MLP − LogisticRegression; aynı 1.000 yeniden örneklem). Kaynak: `results/test_bootstrap_diff.csv`.

| Metrik | Fark | %95 GA | Fark ≤ 0 olan örneklem oranı |
|---|---|---|---|
| AP | +0,016 | [−0,014; +0,041] | 0,19 |
| ROC-AUC | +0,007 | [−0,009; +0,022] | 0,18 |
| F1 | −0,013 | [−0,062; +0,031] | 0,74 |

**Tablo 10.** Test kümesindeki gerçek pozitifler içinde `poutcome` ve `contact` gruplarına göre kaçırılma (FN) oranı.

| Grup | Pozitif sayısı | FN | FN oranı |
|---|---|---|---|
| poutcome = success | 22 | 2 | %9 |
| poutcome = failure | 17 | 17 | %100 |
| poutcome = nonexistent | 51 | 51 | %100 |
| contact = cellular | 74 | 55 | %74 |
| contact = telephone | 16 | 15 | %94 |

**Tablo 11.** Aynı yöntemin üç farklı değerlendirme düzeyinde AP değeri.

| Düzey | Manuel arama / M2 | LogisticRegression |
|---|---|---|
| Development iç CV (seçim skoru, M2) | 0,441 ± 0,046 | – |
| Dış CV (n = 3, Mean ± SD) | 0,429 ± 0,055 | 0,420 ± 0,028 |
| Bağımsız test [%95 GA] | 0,406 [0,313; 0,519] | 0,390 [0,300; 0,509] |
