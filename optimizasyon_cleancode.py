# -*- coding: utf-8 -*-
"""
FLO Showroom Optimizasyon Modeli
Pyomo + Gurobi çözümü (Temizlenmiş Versiyon - Hata Düzeltildi)

@author: HACER
"""
# ============================
# 📦 1. KÜTÜPHANELER
# ============================
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeIntegers, maximize, SolverFactory, value
)
import sys

# ============================
# 📂 2. VERİYİ YÜKLE VE HAZIRLA
# ============================
try:
    data = pd.read_csv(
        r"C:\Users\HACER\OneDrive\Masaüstü\forecast_FW26_results_bestmodel.csv",
        sep=";",
        encoding="utf-8"
    )
except FileNotFoundError:
    print("HATA: Girdi verisi bulunamadı. Lütfen dosya yolunu kontrol edin.")
    sys.exit() # Dosya yoksa dur

print(data.dtypes)
print(data.head())

# ============================
# ⚙️ 3. MODEL PARAMETRELERİ
# ============================

# --- Veriden Gelen Parametreler ---
index_set = data.index.tolist()
p = data["TVALL_Sales_Qty"].to_dict() # Tahmini satış
m = data["Margin"].to_dict()             # Marj
c = data["ListPrice"].to_dict()         # Fiyat (Maliyet)
st = data["Sell_Through"].to_dict()     # Sell-through
line_type = data["Line"].to_dict()      # Basic/Line
channel = data["Channel"].to_dict()     # Kanal

# --- İş Kuralları (Stratejik Parametreler) ---
ANA_BUTCE = 20_000_000
OTB_PAYI_YUZDE = 90.0
BASIC_MIN_ORAN = 0.40
BASIC_MAX_ORAN = 0.60
MARGIN_MIN_ORAN = 0.33
SELL_THROUGH_CARPAN = 500  # beta
CHANNEL_SHARES = {0: 0.6, 1: 0.4} # {Kanal_ID: Pay}

# ============================
# 🧩 4. MODELİ OLUŞTUR
# ============================
model = ConcreteModel("Showroom_Optimization")

# ============================
# 🎯 5. KARAR DEĞİŞKENİ
# ============================
model.x = Var(index_set, domain=NonNegativeIntegers)

# ============================
# 🧮 6. AMAÇ FONKSİYONU
# ============================
def obj_rule(model):
    return sum(p[i] * model.x[i] for i in index_set)
model.objective = Objective(rule=obj_rule, sense=maximize)

# ============================
# ⛓️ 7. KISITLAR
# ============================
print("Model kısıtları oluşturuluyor...")

# (1) Bütçe Kısıtı
def budget_rule(model):
    usable_budget = ANA_BUTCE * (OTB_PAYI_YUZDE / 100.0)
    return sum(c[i] * model.x[i] for i in index_set) <= usable_budget
model.BudgetConstraint = Constraint(rule=budget_rule)

# (2) Basic-Line Oranı Kısıtı
basic_indices = [i for i in index_set if str(line_type[i]).lower() == "basic"]
line_indices = [i for i in index_set if str(line_type[i]).lower() == "line"]
all_indices = basic_indices + line_indices

def basic_min_rule(model):
    return sum(model.x[i] for i in basic_indices) >= BASIC_MIN_ORAN * sum(model.x[i] for i in all_indices)
model.BasicMinConstraint = Constraint(rule=basic_min_rule)

def basic_max_rule(model):
    return sum(model.x[i] for i in basic_indices) <= BASIC_MAX_ORAN * sum(model.x[i] for i in all_indices)
model.BasicMaxConstraint = Constraint(rule=basic_max_rule)

# (3) Sell-Through (Devir) Kısıtı
def sellthrough_rule(model, i):
    return model.x[i] <= st[i] * SELL_THROUGH_CARPAN
model.SellThroughConstraint = Constraint(index_set, rule=sellthrough_rule)

# (4) Range Plan Kısıtı (Tahminden fazla SKU atama)
def range_plan_rule(model, i):
    return model.x[i] <= p[i]
model.RangePlanConstraint = Constraint(index_set, rule=range_plan_rule)

# (5) Kanal Payı Kısıtı
total_x_expr = sum(model.x[i] for i in index_set)
for ch, share in CHANNEL_SHARES.items():
    indices = [i for i in index_set if channel[i] == ch]
    model.add_component(
        f"ChannelShare_{ch}",
        Constraint(expr = sum(model.x[i] for i in indices) == share * total_x_expr)
    )

# (6) Ortalama Marj Kısıtı
# === HATALI 'IF' KONTROLÜ DÜZELTİLDİ (SİZİN ORİJİNAL HALİNE DÖNÜLDÜ) ===
def avg_margin_rule(model):
    # Bu, (sum(m[i]*x[i]) / sum(x[i])) >= MARGIN_MIN_ORAN 
    # ifadesinin lineer (doğrusal) halidir ve sıfıra bölme riski taşımaz.
    return sum(m[i] * model.x[i] for i in index_set) >= MARGIN_MIN_ORAN * sum(model.x[i] for i in index_set)
model.AvgMarginConstraint = Constraint(rule=avg_margin_rule)
# ====================================================================

# ============================
# 🚀 8. MODELİ ÇÖZ
# ============================
print("Model Gurobi solver ile çözülüyor...")
solver = SolverFactory("gurobi")
results = solver.solve(model, tee=False) # Logları görmek için tee=True yapabilirsiniz

# Çözüm durumunu kontrol et
if (results.solver.status != 'ok') or (results.solver.termination_condition != 'optimal'):
    print(f"HATA: Model optimal bir çözüm bulamadı. Durum: {results.solver.termination_condition}")
    sys.exit()
    
print("✅ Model optimal olarak çözüldü.")

# ============================
# 📊 9. SONUÇLARI KAYDET
# ============================
data["Optimal_SKU_FW26"] = [round(value(model.x[i])) for i in index_set]
output_path = "optimization_results_FW26_pyomoyeni.csv"
data.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
print(f"💾 Sonuçlar şuraya kaydedildi: {output_path}")

# ============================
# 📈 10. OPTİMİZASYON VALİDASYON RAPORU
# ============================
total_sku = data["Optimal_SKU_FW26"].sum()

if total_sku == 0:
    print("\nUYARI: Model optimal SKU ataması yapmadı (Toplam SKU = 0). Kısıtlar çok sıkı olabilir.")
    sys.exit()

# --- KPI Hesaplamaları ---
total_budget_used = (data["Optimal_SKU_FW26"] * data["ListPrice"]).sum()
usable_budget = ANA_BUTCE * (OTB_PAYI_YUZDE / 100.0)
budget_util_percent = (total_budget_used / usable_budget) * 100
avg_margin_realized = (sum(data["Margin"] * data["Optimal_SKU_FW26"]) / total_sku) * 100

# --- Raporu Yazdır ---
print("\n" + "="*50)
print("📈 OPTİMİZASYON VALİDASYON RAPORU 📈")
print("="*50)

print(f"\n🎯 Amaç Fonksiyonu Değeri (Toplam Potansiyel): {value(model.objective):,.0f}")
print(f"📦 Toplam Optimal SKU Adedi: {total_sku:,.0f}")

print("\n--- 💰 Bütçe Kısıtı Performansı ---")
print(f"  Kullanılabilir Bütçe: {usable_budget:,.0f} TL  ({OTB_PAYI_YUZDE:.0f}%)")
print(f"  Kullanılan Bütçe:   {total_budget_used:,.0f} TL")
print(f"  BÜTÇE KULLANIMI:    {budget_util_percent:.2f}%")

print("\n--- 📊 Stratejik Kısıtların Performansı ---")
margin_target = MARGIN_MIN_ORAN * 100
print(f"  Ortalama Marj:   {avg_margin_realized:.2f}% (Hedef: Min {margin_target:.2f}%)")

print("\n  Kanal Payları:")
for ch, share in CHANNEL_SHARES.items():
    channel_sum = data[data["Channel"] == ch]["Optimal_SKU_FW26"].sum()
    realized_share = (channel_sum / total_sku) * 100
    target_share = share * 100
    print(f"    - Kanal {ch}: {realized_share:.1f}% (Hedef: {target_share:.1f}%)")

print("\n" + "="*50)