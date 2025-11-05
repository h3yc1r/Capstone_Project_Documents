# 🧠 Range Planı Optimizasyon Projesi

> Geçmiş sezon satış verilerini kullanarak FW26 sezonu için satış tahmini ve üretim optimizasyonu yapan uçtan uca karar destek sistemi.

---

## 📌 Proje Özeti

Bu proje, *Bir perakende şirketinin range planı ürün planlaması* sürecini desteklemek amacıyla geliştirilmiştir.  
Amaç, geçmiş dört sezonun (FW22–FW25) satış verilerini kullanarak *FW26 sezonu için satış tahmini yapmak* ve bu tahminleri *optimizasyon modeli* aracılığıyla kullanarak hangi SKU’ların hangi miktarda üretilmesi gerektiğini belirlemektir.

Proje üç temel aşamadan oluşmaktadır:
1. *Veri Hazırlama ve Satış Tahmini (Forecast)*
2. *Üretim Optimizasyonu (Pyomo + Glpk)*
3. *Sonuçların Görselleştirilmesi (Streamlit)*

---

## 🧩 Dosya Yapısı

| Dosya Adı | Açıklama |
|------------|-----------|
| 🐍 LAST_FORECAST.py | FW22–FW25 geçmiş satış verilerini kullanarak FW26 sezonu için satış tahmini yapan ana Python dosyası. |
| 🐍 rangeoptimizasyon_code.py | Pyomo + Glpk ile oluşturulmuş optimizasyon modeli. Tahmin sonuçlarını kullanarak üretim miktarlarını maksimize eder. |
| 🐍 app.py | Streamlit arayüzü — tahmin ve optimizasyon çıktılarının görselleştirilmesini sağlar. |
| 📄 forecast_input_FW26_güncel_with_newcols.csv | FW26 sezonu için tahmin modeline girilen, özellik mühendisliği uygulanmış güncel veri seti. |
| 📄 optimization_results_FW26_pyomoyeni.csv | Optimizasyon modelinden elde edilen sonuçlar — her SKU için önerilen üretim miktarlarını içerir. |
| 📄 training_data_FW22_FW25_güncel_with_newcols.csv | FW22–FW25 sezonlarına ait geçmiş satış verilerini içeren ve tahmin modelinin eğitiminde kullanılan güncel veri seti. |
| 🧾 requirements.txt | Projenin çalışması için gereken Python kütüphanelerinin listesi. |
| 🧩 packages.txt | Streamlit Cloud veya sanal ortam kurulumu için alternatif bağımlılık listesi. |
| 🖼 flo_logo.png | Arayüzde kullanılan FLO logosu görseli. |
| 📄 README.md | Bu dokümantasyon dosyası. Projenin genel açıklamasını ve dosya yapısını içerir. |

---

## ⚙ Metodoloji

### ⿡ Satış Tahmini (Forecasting)
- *Girdi:* FW22–FW25 sezonlarına ait satış verileri  
- *Çıktı:* FW26 sezonu satış tahminleri  
- *Kullanılan yöntemler:* Özellik mühendisliği, model karşılaştırma ve en iyi model seçimi  
- *Değerlendirme metrikleri:* MAPE, RMSE, R²  

### ⿢ Optimizasyon Modeli
- *Araçlar:* Pyomo + Glpk  
- *Amaç:* Toplam beklenen karı maksimize etmek  
- *Karar değişkeni:* SKU bazında üretilecek miktar  
- *Kısıtlar:* Stok sınırları, kategori oranları, üretim kapasitesi  

### ⿣ Görselleştirme
- *Araç:* Streamlit  
- *Amaç:* Tahmin ve optimizasyon sonuçlarını kullanıcı dostu bir arayüzde sunmak  

---

## 💻 Streamlit Dashboard Özeti

Bu web arayüzü, geçmiş sezon verilerini kullanarak *FW26 sezonu için satış tahmini* ve *optimum üretim planı* oluşturan etkileşimli bir *karar destek sistemidir.*

---

### 🧠 Talep Tahmini

- FW22–FW25 verilerini analiz edip FW26 satışlarını tahmin eder.  
- En iyi modeli (*Linear Regression, **Random Forest, **XGBoost*) otomatik olarak seçer.  
- Kullanıcı, global indirim oranını değiştirerek *senaryo simülasyonu* yapabilir.  

*Sunulan çıktılar:*
- 📊 Toplam satış tahmini  
- 📈 Geçen sezona göre büyüme oranı  
- 🎨 Marka, klasman ve sezon bazlı görsel kırılımlar  
- 💾 Sonuçları CSV olarak indirme imkânı  

---

### 📈 Optimizasyon

- Tahmin sonuçlarını kullanarak *bütçe, **marj* ve *ürün oranı* kısıtları altında en iyi üretim planını oluşturur.  

*Parametreler:*
💰 Bütçe • 📊 Marj • 🎨 Basic/Line oranı • 📦 OTB payı  

*Sunulan çıktılar:*
- 🎯 Toplam satış potansiyeli  
- 💰 Kullanılan bütçe oranı  
- 📈 Ortalama marj ve Basic oranı  
- 📊 Marka, klasman ve line bazlı grafik dağılımlar  
- 💾 İndirilebilir optimal plan dosyası  

---

## 🧮 Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|----------|------------|
| Tahminleme | Python (Pandas, Scikit-learn, XGBoost) |
| Optimizasyon | Pyomo, Glpk |
| Görselleştirme | Streamlit |
| Veri İşleme | Pandas, CSV |
| Versiyon Kontrolü | Git + GitHub |

---

> ⚠ Bu projede kullanılan tüm veriler *tamamen sentetik olarak üretilmiştir* ve *gerçek ticari verileri yansıtmamaktadır.*  
Veriler yalnızca *akademik amaçlarla (Capstone Projesi)* kullanılmıştır.
