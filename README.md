# 🧠 Showroom Optimizasyon Projesi

> Geçmiş sezon satış verilerini kullanarak FW26 sezonu için satış tahmini ve üretim optimizasyonu yapan uçtan uca karar destek sistemi.

---

## 📌 Proje Özeti

Bu proje, **Bir perakende şirketinin showroom ürün planlaması** sürecini desteklemek amacıyla geliştirilmiştir.  
Amaç, geçmiş dört sezonun (FW22–FW25) satış verilerini kullanarak **FW26 sezonu için satış tahmini yapmak** ve bu tahminleri **optimizasyon modeli** aracılığıyla kullanarak hangi SKU’ların hangi miktarda üretilmesi gerektiğini belirlemektir.

Proje üç temel aşamadan oluşmaktadır:
1. **Veri Hazırlama ve Satış Tahmini (Forecast)**
2. **Üretim Optimizasyonu (Pyomo + Gurobi)**
3. **Sonuçların Görselleştirilmesi (Streamlit - yakında eklenecek)**

---

## 🧩 Dosya Yapısı

| Dosya Adı | Açıklama |
|------------|-----------|
| 🐍 `LASTFLOFORECASTuntitled0.py` | Ana tahminleme (forecast) dosyası. Geçmiş verilerle FW26 sezonu satışlarını makine öğrenmesi modelleriyle tahmin eder. |
| 📄 `forecast_FW26_results_bestmodel.csv` | En iyi performans gösteren tahmin modelinden elde edilen FW26 satış tahmin sonuçları. |
| 📄 `forecast_input_FW26_güncel.csv` | FW26 sezonu için tahminleme modeline girilen, özellik mühendisliği uygulanmış veri seti. |
| 🐍 `optimizasyon_cleancode.py` | Pyomo + Gurobi kullanılarak oluşturulmuş temizlenmiş optimizasyon modeli. Tahmin sonuçlarına göre üretim miktarlarını optimize eder. |
| 📄 `optimization_results_FW26_pyomoyeni.csv` | Optimizasyon modelinin çıktı dosyası. Her SKU için önerilen üretim miktarlarını içerir. |
| 📄 `training_data_FW22_FW25_güncel.csv` | FW22–FW25 arası geçmiş satış verilerini içeren, tahmin modelinin eğitiminde kullanılan veri seti. |
| 📄 `README.md` | Bu dokümantasyon dosyası. Projenin genel açıklamasını ve dosya yapısını içerir. |
| 🌐 *(yakında)* `streamlit_app.py` | Tahmin ve optimizasyon sonuçlarını etkileşimli olarak görselleştiren Streamlit uygulaması. |

---

## ⚙️ Metodoloji

### 1️⃣ Satış Tahmini (Forecasting)
- **Girdi:** FW22–FW25 sezonlarına ait satış verileri  
- **Çıktı:** FW26 sezonu satış tahminleri  
- **Kullanılan yöntemler:** Özellik mühendisliği, model karşılaştırma ve en iyi model seçimi  
- **Değerlendirme metrikleri:** MAPE, RMSE, R²  

### 2️⃣ Optimizasyon Modeli
- **Araçlar:** Pyomo + Gurobi  
- **Amaç:** Toplam beklenen karı maksimize etmek  
- **Karar değişkeni:** SKU bazında üretilecek miktar  
- **Kısıtlar:** Stok sınırları, kategori oranları, üretim kapasitesi  

### 3️⃣ Görselleştirme (Yakında)
- **Araç:** Streamlit  
- **Amaç:** Tahmin ve optimizasyon sonuçlarını kullanıcı dostu bir arayüzde sunmak  

---

## 🧮 Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|----------|------------|
| Tahminleme | Python (Pandas, Scikit-learn, XGBoost) |
| Optimizasyon | Pyomo, Gurobi |
| Görselleştirme | Streamlit |
| Veri İşleme | Pandas, CSV |
| Versiyon Kontrolü | Git + GitHub |

---

## ▶️ Çalıştırma Adımları

1. **Depoyu klonla**
   ```bash
   git clone https://github.com/<kullanıcı-adın>/<repo-adı>.git
   cd <repo-adı>
