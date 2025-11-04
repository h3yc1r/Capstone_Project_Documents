# ===============================================
# 📦 1. GEREKLİ KÜTÜPHANELER
# ===============================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

print("✅ Kütüphaneler yüklendi.")


# ===============================================
# 📂 2. VERİLERİ YÜKLE
# ===============================================
train_path = r"C:\Users\fatma\OneDrive\Masaüstü\training_data_FW22_FW25_güncel_with_newcols.csv"
test_path  = r"C:\Users\fatma\OneDrive\Masaüstü\forecast_input_FW26_güncel_with_newcols.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# ===============================================
# 🔍 3. GÖRSEL EDA (KEŞİFSEL VERİ ANALİZİ)
# ===============================================

# 1️⃣ Sezon bazlı ortalama satış trendi
plt.figure(figsize=(8,5))
season_sales = train.groupby("Season")["TVALL_Sales_Qty"].mean().reset_index()
sns.lineplot(data=season_sales, x="Season", y="TVALL_Sales_Qty", marker="o")
plt.title("📆 Sezonlara Göre Ortalama Satış Trendi")
plt.xticks(rotation=45)
plt.show()

# 2️⃣ Fiyat ve satış ilişkisi
sns.scatterplot(data=train, x="ListPrice", y="TVALL_Sales_Qty", alpha=0.6)
plt.title("💸 Fiyat ve Satış İlişkisi")
plt.show()

# 3️⃣ İndirim oranı ve satış ilişkisi
sns.scatterplot(data=train, x="DiscountRate", y="TVALL_Sales_Qty", alpha=0.6, color="orange")
plt.title("🏷️ İndirim Oranı ve Satış İlişkisi")
plt.show()

# 4️⃣ Stok seviyesi ve satış ilişkisi
sns.lmplot(data=train, x="Stock_Level", y="TVALL_Sales_Qty", height=6, aspect=1.2, line_kws={"color": "red"})
plt.title("📦 Stok Seviyesi ile Satış İlişkisi")
plt.show()

# 5️⃣ Korelasyon Isı Haritası
plt.figure(figsize=(10, 8))
corr = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔥 Korelasyon Isı Haritası (Sayısal Değişkenler)")
plt.show()


# ===============================================
# ⚙️ 4. VERİ HAZIRLIK
# ===============================================
target = "TVALL_Sales_Qty"
categorical_cols = ["Brand", "Gender", "Klasman", "SubCategory", "Line", "Season"]

# Kategorik değişkenleri encode et
X = pd.get_dummies(train.drop(columns=["SKU_ID", target], errors="ignore"),
                   columns=categorical_cols, drop_first=True)
y = train[target]

test_encoded = pd.get_dummies(test.drop(columns=["SKU_ID"], errors="ignore"),
                              columns=categorical_cols, drop_first=True)

# Eksik sütunları hizala
missing_cols = set(X.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0
test_encoded = test_encoded[X.columns]


# ===============================================
# 🧠 5. MODEL LİSTESİ (Denenecek Modeller)
# ===============================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=300, max_depth=10),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6, verbosity=0)
}


# ===============================================
# 🔁 6. MODEL KARŞILAŞTIRMASI (5-Fold Cross Validation)
# ===============================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

results = []
print("\n🔁 5-Fold Cross Validation Sonuçları:\n")

for name, model in models.items():
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring=mae_scorer)
    r2_scores  = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)
    results.append((name, mae_scores.mean(), r2_scores.mean()))
    print(f"{name}: MAE={mae_scores.mean():.2f} | R²={r2_scores.mean():.3f}")


# ===============================================
# 🏆 7. EN İYİ MODELİ SEÇ ve EĞİT
# ===============================================
best_model_name, best_mae, best_r2 = sorted(results, key=lambda x: x[1])[0]
best_model = models[best_model_name]

print(f"\n🏆 En iyi model: {best_model_name}")
print(f"📉 Ortalama MAE: {best_mae:.2f}")
print(f"📈 Ortalama R² : {best_r2:.3f}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_val)
y_pred = np.maximum(np.round(y_pred), 0).astype(int)

print("\n📊 Doğrulama Sonuçları (Validation Set):")
print(f"MAE: {mean_absolute_error(y_val, y_pred):.2f}")
print(f"R² : {r2_score(y_val, y_pred):.3f}")


# ===============================================
# 💾 8. TAHMİN YAP ve KAYDET
# ===============================================
test["TVALL_Sales_Qty"] = np.maximum(np.round(best_model.predict(test_encoded)), 0).astype(int)

output_path = r"C:\Users\fatma\OneDrive\Masaüstü\forecast_FW26_results_bestmodel.csv"
test.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')

print(f"\n✅ Tahminler kaydedildi: {output_path}")
print(f"🧠 Tahmin modeli: {best_model_name}")
print(f"🧮 TVALL_Sales_Qty sütunu güncellendi (tamsayı, negatif yok).")


# ===============================================
# 🧠 9. MODELİN ÖĞRENDİKLERİ (Feature Importance)
# ===============================================
if best_model_name == "XGBoost":
    importance = best_model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10,6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title("🧠 XGBoost Özellik Önem Grafiği (İlk 15 Değişken)")
    plt.show()
    
    # ===============================================
# 📊 MODEL KARŞILAŞTIRMA GRAFİĞİ (MAE ve R²)
# ===============================================
results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2"])

fig, ax1 = plt.subplots(figsize=(8,5))
sns.barplot(data=results_df, x="Model", y="MAE", ax=ax1, color="lightblue", label="MAE")
ax2 = ax1.twinx()
sns.lineplot(data=results_df, x="Model", y="R2", ax=ax2, color="darkorange", marker="o", label="R²")

ax1.set_title("📊 Model Performans Karşılaştırması (MAE vs R²)")
ax1.set_ylabel("MAE (Hata)")
ax2.set_ylabel("R² (Açıklama Gücü)")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()



