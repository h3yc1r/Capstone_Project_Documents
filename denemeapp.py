# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 19:00:35 2025

@author: fatma
"""

# ===============================================
# 📦 1. GEREKLİ KÜTÜPHANELER
# ===============================================
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import sys
import plotly.express as px
import plotly.graph_objects as go

# Makine Öğrenmesi Kütüphaneleri
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 

# Optimizasyon Kütüphaneleri
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeIntegers, maximize, SolverFactory, value
)

warnings.filterwarnings("ignore")

# ===============================================
# 🎨 2. SAYFA YAPILANDIRMASI VE STİL
# ===============================================
st.set_page_config(
    page_title="Karar Destek Aracı",
    page_icon="💎", # Sayfa sekmesi ikonu
    layout="wide"
)

# Modern ve kurumsal bir görünüm için özel CSS
st.markdown("""
<style>
    /* Ana font */
    html, body, [class*="st-"] {
        font-family: 'Roboto', 'Inter', sans-serif;
    }
    
    /* Sidebar stili */
    .css-18e3th9 {
        background-color: #f5f5f5; /* Açık gri sidebar */
    }
    
    /* Sekme stilleri (Talep Tahmini sayfası için) */
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #0068c9; /* Mavi vurgu rengi */
    }
    
    /* Metrik KPI kartları */
    .css-1b3wcvb {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        color: #004a91; /* Kurumsal mavi tonu */
    }
</style>
""", unsafe_allow_html=True)


# ===============================================
# 📂 3. SABİT DOSYA YOLLARI
# ===============================================
# Bu yolların DOĞRU olduğundan emin olun
try:
    # --- Forecast Girdileri ---
    TRAIN_PATH = "training_data_FW22_FW25_güncel_with_newcols.csv"
    TEST_PATH = "forecast_input_FW26_güncel_with_newcols.csv"

    # --- Optimizasyon Girdisi (Forecast'in çıktısı) ---
    OPTIMIZATION_INPUT_PATH = "forecast_FW26_results_bestmodel.csv"
except Exception as e:
    st.error(f"Dosya yolları tanımlanırken bir hata oluştu: {e}")
    st.stop()
    
# ===============================================
# ⚡ 4. HIZLANDIRILMIŞ (CACHED) FONKSİYONLAR
# ===============================================

# --- Global Veri Yükleme ---
@st.cache_data
def load_all_data(train_path, test_path, opt_path):
    """Tüm gerekli verileri tek seferde yükler."""
    train_df, test_df, opt_input_df = None, None, None
    try:
        train_df = pd.read_csv(train_path)
    except FileNotFoundError:
        st.sidebar.error(f"Eğitim verisi bulunamadı:\n{train_path}")
        
    try:
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        st.sidebar.error(f"Tahmin (input) verisi bulunamadı:\n{test_path}")

    try:
        opt_input_df = pd.read_csv(opt_path, sep=";", encoding="utf-8")
    except FileNotFoundError:
        st.sidebar.warning(f"Optimizasyon verisi ({opt_path.split(chr(92))[-1]}) bulunamadı. Lütfen önce Talep Tahmini modelini çalıştırın.")
        
    return train_df, test_df, opt_input_df

# --- Global Yardımcılar ---
@st.cache_data
def convert_df_to_csv(df):
    """DataFrame'i CSV formatında byte'a çevirir."""
    return df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')

# --- Forecast Sayfası Yardımcıları ---
@st.cache_data
def preprocess_data(train, test):
    """Veriyi işler, encode eder ve model için hazırlar."""
    target = "TVALL_Sales_Qty"
    categorical_cols = ["Brand", "Gender", "Klasman", "SubCategory", "Line", "Season"]
    X = pd.get_dummies(train.drop(columns=["SKU_ID", target], errors="ignore"),
                       columns=categorical_cols, drop_first=True)
    y = train[target]
    test_processed = pd.get_dummies(test.drop(columns=["SKU_ID"], errors="ignore"),
                                    columns=categorical_cols, drop_first=True)
    missing_cols = set(X.columns) - set(test_processed.columns)
    for col in missing_cols: test_processed[col] = 0
    test_processed = test_processed[X.columns]
    return X, y, test_processed

@st.cache_data
def run_model_comparison(X, y):
    """Modelleri 5-Fold CV ile karşılaştırır."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=6, verbosity=0)
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42); mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False); r2_scorer = make_scorer(r2_score)
    results = []
    for name, model in models.items():
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring=mae_scorer, n_jobs=-1)
        r2_scores  = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer, n_jobs=-1)
        results.append((name, mae_scores.mean(), r2_scores.mean()))
    results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2"]).sort_values("MAE")
    return results_df, models

@st.cache_resource
def train_best_model(X, y, _models, best_model_name):
    """En iyi modeli eğitir ve eğitilmiş modeli döndürür."""
    best_model = _models[best_model_name]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)
    y_pred = np.maximum(np.round(y_pred), 0).astype(int)
    val_mae = mean_absolute_error(y_val, y_pred)
    val_r2 = r2_score(y_val, y_pred)
    return best_model, X_val, y_val, y_pred, val_mae, val_r2

@st.cache_data
def get_feature_importance(_model, _columns): # UnhashableParamError için düzeltildi
    """Modelin özellik önemini alır."""
    if hasattr(_model, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': _columns, 'Importance': _model.feature_importances_})
        return importance_df.sort_values(by='Importance', ascending=False).head(15)
    return None

# ===============================================
# 🏁 5. UYGULAMA BAŞLANGICI VE VERİ YÜKLEME
# ===============================================

# Session state'i başlat
if 'data_loaded' not in st.session_state:
    train_df, test_df, opt_input_df = load_all_data(TRAIN_PATH, TEST_PATH, OPTIMIZATION_INPUT_PATH)
    if train_df is not None and test_df is not None:
        st.session_state.train_df = train_df
        st.session_state.test_df = test_df
        st.session_state.opt_input_df = opt_input_df # Bu, optimizasyonun girdisidir
        st.session_state.data_loaded = True
        st.sidebar.success("Tüm veriler başarıyla yüklendi.")
    else:
        st.sidebar.error("Ana veriler yüklenemedi. Lütfen dosya yollarını kontrol edin.")
        st.stop()


# ===============================================
# ⬅️ 6. KENAR ÇUBUĞU (SIDEBAR) NAVİGASYONU
# ===============================================

# Logo Ekleme (Yerel dosyadan)
try:
    # 'use_column_width' -> 'use_container_width' olarak düzeltildi (Uyarı için)
    st.sidebar.image("flo_logo.png", use_container_width=True) 
except Exception as e:
    st.sidebar.warning(f"Logo yüklenemedi (flo_logo.png bulunamadı).")
    st.sidebar.image("https://placehold.co/400x100/004a91/ffffff?text=LOGO", use_container_width=True)

st.sidebar.title("Ana Navigasyon")

# Sayfa Sıralaması Değişikliği
page = st.sidebar.radio(
    "Gitmek istediğiniz sayfayı seçin:",
    ["🧠 Talep Tahmini (Analist Modeli)", "📈 Optimizasyon (Karar Modeli)"], # Sıralama değişti
    label_visibility="collapsed"
)

st.sidebar.divider()

# --- Navigasyona Göre Değişen Sidebar Widget'ları ---

if page == "📈 Optimizasyon (Karar Modeli)":
    # ---------------------------------
    # OPTİMİZASYON KONTROL PANELİ
    # ---------------------------------
    st.sidebar.header("⚙️ Optimizasyon Parametreleri")
    
    Ana_Butce_input = st.sidebar.number_input(
        "Ana Bütçe (TL)", 
        min_value=1_000_000, 
        value=20_000_000, 
        step=1_000_000,
        help="Planlamaya ayrılan toplam bütçe."
    )
    OTB_Payi_input = st.sidebar.slider(
        "OTB Kullanım Payı (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0, # Varsayılan %90
        step=1.0,
        help="Ana bütçenin yüzde kaçının bu optimasyonda kullanılacağı."
    )
    basic_ratio_input = st.sidebar.slider(
        "Basic Oran Aralığı (%)",
        min_value=0.0,
        max_value=100.0,
        value=(40.0, 60.0),
        step=1.0,
        help="Toplam SKU'lar içindeki 'Basic' ürünlerin minimum ve maksimum yüzdesi."
    )
    margin_min_input = st.sidebar.slider(
        "Minimum Ortalama Marj (%)",
        min_value=0.0,
        max_value=100.0,
        value=33.0, # Varsayılan %33
        step=0.5,
        help="Tüm planın ortalama marjı en az bu değer olmalı."
    )
    
elif page == "🧠 Talep Tahmini (Analist Modeli)":
    # ---------------------------------
    # FORECAST KONTROL PANELİ
    # ---------------------------------
    st.sidebar.header("🏷 Segment Filtreleri")
    st.sidebar.caption("Aşağıdaki filtreler 'Ürün/Marka Kırılımı' sekmesini etkiler.")
    
    brand_list = ["Tümü"] + sorted(st.session_state.test_df["Brand"].unique().tolist())
    gender_list = ["Tümü"] + sorted(st.session_state.test_df["Gender"].unique().tolist())
    klasman_list = ["Tümü"] + sorted(st.session_state.test_df["Klasman"].unique().tolist())
    
    st.session_state.filter_brand = st.sidebar.selectbox("Marka Seçin", brand_list)
    st.session_state.filter_gender = st.sidebar.selectbox("Cinsiyet Seçin", gender_list)
    st.session_state.filter_klasman = st.sidebar.selectbox("Klasman Seçin", klasman_list)
    
    st.sidebar.divider()
    
    st.sidebar.header("🔄 Senaryo Oluşturucu")
    st.sidebar.caption("Modelin farklı senaryolara tepkisini ölçün. (Bu, 'Genel Özet' sekmesini etkiler)")
    
    st.session_state.discount_change = st.sidebar.slider("Global İndirim Oranı Değişimi (%)", -20.0, 20.0, 0.0, 0.5)


# ===============================================
# 📑 7. ANA EKRAN (SAYFA GÖSTERİMİ)
# ===============================================

if page == "🧠 Talep Tahmini (Analist Modeli)":
    
    # ====================================================
    # SAYFA 1: TALEP TAHMİNİ (ANALİST DASHBOARD'U)
    # ====================================================

    st.title("🧠 Talep Tahmini (Analist Modeli)")
    
    if 'data_loaded' not in st.session_state or st.session_state.train_df is None or st.session_state.test_df is None:
        st.error("Tahmin modelini çalıştırmak için 'training_data' ve 'forecast_input' verileri yüklenemedi. Lütfen dosya yollarını kontrol edin.")
        st.stop()

    st.warning("Bu sayfa, model eğitim sürecini gösterir. Yeni bir tahmin dosyası oluşturmak ve 'Optimizasyon' sayfasını güncellemek için aşağıdaki butonu kullanın.")
    
    if st.button("Modeli Yeniden Eğit ve FW26 Tahminlerini Kaydet", type="primary", key="run_forecast"):
        
        try:
            with st.spinner("Adım 1/5: Veri hazırlanıyor ve ön işleniyor..."):
                X, y, test_encoded = preprocess_data(st.session_state.train_df, st.session_state.test_df)
            
            with st.spinner("Adım 2/5: Modeller 5-Fold Cross-Validation ile karşılaştırılıyor..."):
                results_df, models = run_model_comparison(X, y)
                best_model_name = results_df.iloc[0]["Model"]
            
            with st.spinner(f"Adım 3/5: 🏆 En iyi model ({best_model_name}) eğitiliyor..."):
                best_model, X_val, y_val, y_pred_val, val_mae, val_r2 = train_best_model(X, y, models, best_model_name)
            
            with st.spinner("Adım 4/5: FW26 sezonu için final tahminleri yapılıyor..."):
                fw26_predictions = best_model.predict(test_encoded)
                fw26_predictions_clean = np.maximum(np.round(fw26_predictions), 0).astype(int)
                
                test_output_df = st.session_state.test_df.copy()
                test_output_df["TVALL_Sales_Qty"] = fw26_predictions_clean
            
            with st.spinner(f"Adım 5/5: Tahminler {OPTIMIZATION_INPUT_PATH.split(chr(92))[-1]} dosyasına kaydediliyor..."):
                test_output_df.to_csv(OPTIMIZATION_INPUT_PATH, index=False, sep=';', encoding='utf-8-sig')
                
                st.session_state.opt_input_df = test_output_df
                
            st.success(f"✅ Başarılı! {OPTIMIZATION_INPUT_PATH} dosyası güncellendi. 'Optimizasyon' sayfası artık bu yeni tahminleri kullanabilir.")
            st.balloons()
            
            # Yeni eğitim sonrası state'i de güncelle
            st.session_state.results_df = results_df
            st.session_state.best_model_name = best_model_name
            st.session_state.best_model = best_model
            st.session_state.X_val, st.session_state.y_val, st.session_state.y_pred_val = X_val, y_val, y_pred_val
            st.session_state.val_mae, st.session_state.val_r2 = val_mae, val_r2
            st.session_state.X_columns = X.columns
            st.session_state.test_output_df = test_output_df
            last_season_name = sorted(st.session_state.train_df["Season"].unique())[-1]
            st.session_state.last_season_name = last_season_name
            st.session_state.fw26_forecast_sum_base = test_output_df["TVALL_Sales_Qty"].sum()
            st.session_state.fw_actuals_sum = st.session_state.train_df[st.session_state.train_df["Season"] == last_season_name]["TVALL_Sales_Qty"].sum()

        except Exception as e:
            st.error(f"Model eğitimi sırasında bir hata oluştu: {e}")
            st.stop()
            
    st.divider()
    
    if 'best_model' not in st.session_state:
        try:
            with st.spinner("Analiz modülü yükleniyor... (İlk çalıştırma)"):
                X, y, test_encoded = preprocess_data(st.session_state.train_df, st.session_state.test_df)
                results_df, models = run_model_comparison(X, y)
                best_model_name = results_df.iloc[0]["Model"]
                best_model, X_val, y_val, y_pred_val, val_mae, val_r2 = train_best_model(X, y, models, best_model_name)
                
                st.session_state.results_df = results_df
                st.session_state.best_model_name = best_model_name
                st.session_state.best_model = best_model
                st.session_state.X_val, st.session_state.y_val, st.session_state.y_pred_val = X_val, y_val, y_pred_val
                st.session_state.val_mae, st.session_state.val_r2 = val_mae, val_r2
                st.session_state.X_columns = X.columns
                
                fw26_predictions = best_model.predict(test_encoded)
                fw26_predictions_clean = np.maximum(np.round(fw26_predictions), 0).astype(int)
                test_output_df = st.session_state.test_df.copy()
                test_output_df["TVALL_Sales_Qty"] = fw26_predictions_clean
                st.session_state.test_output_df = test_output_df
                
                last_season_name = sorted(st.session_state.train_df["Season"].unique())[-1]
                st.session_state.last_season_name = last_season_name
                st.session_state.fw26_forecast_sum_base = test_output_df["TVALL_Sales_Qty"].sum()
                st.session_state.fw_actuals_sum = st.session_state.train_df[st.session_state.train_df["Season"] == last_season_name]["TVALL_Sales_Qty"].sum()

        except Exception as e:
            st.error(f"İlk model eğitim pipeline'ı çalışırken hata oluştu: {e}")
            st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Genel Özet", 
        "🔍 Satış ve Tahmin Analizi", 
        "🏷 Ürün / Marka / Sezon Kırılımı", 
        "⚙ Model Performansı"
    ])

    with tab1:
        st.header("Genel Özet (Executive Summary)")
        
        scenario_test_df = preprocess_data(st.session_state.train_df, st.session_state.test_df)[2].copy()
        original_discount = scenario_test_df["DiscountRate"]
        new_discount = (original_discount * (1 + (st.session_state.discount_change / 100.0))).clip(0, 1) 
        scenario_test_df["DiscountRate"] = new_discount
        scenario_preds = st.session_state.best_model.predict(scenario_test_df)
        scenario_sum = np.maximum(np.round(scenario_preds), 0).astype(int).sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="🧮 Toplam Tahmin Edilen Satış (FW26)", 
            value=f"{scenario_sum:,.0f} Adet",
            delta=f"{(scenario_sum - st.session_state.fw26_forecast_sum_base):,.0f} (Baz Modele Göre)",
            help=f"Baz model tahmini: {st.session_state.fw26_forecast_sum_base:,.0f}"
        )
        growth_delta_scenario = (scenario_sum - st.session_state.fw_actuals_sum) / st.session_state.fw_actuals_sum
        col2.metric(
            label=f"📈 Beklenen Büyüme (vs {st.session_state.last_season_name})", 
            value=f"{growth_delta_scenario:.1%}",
            delta_color="normal"
        )
        col3.metric(
            label="🏆 En İyi Model", 
            value=st.session_state.best_model_name
        )
        
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Tahmin vs Gerçek ({st.session_state.last_season_name})")
            df_compare = pd.DataFrame({
                "Sezon": [st.session_state.last_season_name, f"FW26 (Senaryo: {st.session_state.discount_change}%)"],
                "Satış Adedi": [st.session_state.fw_actuals_sum, scenario_sum]
            })
            fig_bar_compare = px.bar(df_compare, x="Sezon", y="Satış Adedi", text="Satış Adedi", title=f"{st.session_state.last_season_name} Gerçekleşen vs FW26 Tahmin")
            fig_bar_compare.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_bar_compare, use_container_width=True)

        with col2:
            st.subheader("Sezon Bazlı Satış Trendi")
            season_sales = st.session_state.train_df.groupby("Season")["TVALL_Sales_Qty"].sum().reset_index()
            fw26_row = pd.DataFrame([{"Season": f"FW26 (Senaryo: {st.session_state.discount_change}%)", "TVALL_Sales_Qty": scenario_sum}])
            season_sales = pd.concat([season_sales, fw26_row], ignore_index=True)
            fig_line_trend = px.line(season_sales, x="Season", y="TVALL_Sales_Qty", title="Tüm Sezonlar ve FW26 Tahmini Satış Trendi", markers=True)
            fig_line_trend.update_traces(texttemplate='%{y:,.0f}', textposition="top center")
            st.plotly_chart(fig_line_trend, use_container_width=True)

    with tab2:
        st.header("Satış ve Tahmin Analizi")
        st.caption(f"Modelin doğruluk payı, en iyi model ({st.session_state.best_model_name}) seçildikten sonra ayrılan doğrulama (validation) seti üzerinde test edilmiştir.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gerçek vs Tahmin Dağılımı")
            df_val = pd.DataFrame({'Gerçek Satış': st.session_state.y_val, 'Tahmin Edilen Satış': st.session_state.y_pred_val})
            fig_scatter = px.scatter(df_val, x='Gerçek Satış', y='Tahmin Edilen Satış', title='Gerçek vs Tahmin (Validation Set)', opacity=0.5, trendline='ols', trendline_color_override='red')
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.subheader("Model Hata Dağılımı (Histogram)")
            errors = st.session_state.y_val - st.session_state.y_pred_val
            fig_hist = px.histogram(errors, nbins=50, title='Hata Dağılımı (Gerçek - Tahmin)')
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.subheader("Tahmin Güven Aralığı (Simülasyon)")
        st.caption(f"Grafik, modelin ortalama hatasını (MAE: {st.session_state.val_mae:,.0f} adet) kullanarak bir güven aralığı simülasyonu yapar.")
        
        ci_df = pd.DataFrame({'Gerçek Satış': st.session_state.y_val, 'Tahmin Edilen Satış': st.session_state.y_pred_val}).sort_values('Gerçek Satış').reset_index(drop=True)
        ci_df['Alt Sınır (Tahmin - MAE)'] = ci_df['Tahmin Edilen Satış'] - st.session_state.val_mae
        ci_df['Üst Sınır (Tahmin + MAE)'] = ci_df['Tahmin Edilen Satış'] + st.session_state.val_mae
        
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(x=ci_df.index, y=ci_df['Üst Sınır (Tahmin + MAE)'], mode='lines', line=dict(color='rgba(211,211,211,0.5)'), name='Üst Sınır'))
        fig_ci.add_trace(go.Scatter(x=ci_df.index, y=ci_df['Alt Sınır (Tahmin - MAE)'], mode='lines', line=dict(color='rgba(211,211,211,0.5)'), name='Alt Sınır', fill='tonexty', fillcolor='rgba(211,211,0.2)'))
        fig_ci.add_trace(go.Scatter(x=ci_df.index, y=ci_df['Tahmin Edilen Satış'], mode='lines', line=dict(color='orange'), name='Tahmin'))
        fig_ci.add_trace(go.Scatter(x=ci_df.index, y=ci_df['Gerçek Satış'], mode='lines', line=dict(color='#0068c9'), name='Gerçek Satış'))
        fig_ci.update_layout(title='Tahmin Güven Aralığı (Gerçek Satışa Göre Sıralı)', xaxis_title='Data Points (Sıralı)', yaxis_title='Satış Adedi')
        st.plotly_chart(fig_ci, use_container_width=True)

    with tab3:
        st.header("Ürün / Marka / Sezon Kırılımı")
        st.info("Sol menüdeki filtreleri kullanarak FW26 tahminlerini segment bazlı inceleyebilirsiniz.")

        filtered_output_df = st.session_state.test_output_df.copy()
        if st.session_state.filter_brand != "Tümü":
            filtered_output_df = filtered_output_df[filtered_output_df["Brand"] == st.session_state.filter_brand]
        if st.session_state.filter_gender != "Tümü":
            filtered_output_df = filtered_output_df[filtered_output_df["Gender"] == st.session_state.filter_gender]
        if st.session_state.filter_klasman != "Tümü":
            filtered_output_df = filtered_output_df[filtered_output_df["Klasman"] == st.session_state.filter_klasman]

        st.subheader("Filtrelenmiş Toplam Tahmin")
        st.metric("Toplam Tahmin (Filtreli)", f"{filtered_output_df['TVALL_Sales_Qty'].sum():,.0f} Adet")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Marka Bazlı Tahmin Dağılımı")
            df_plot = filtered_output_df.groupby("Brand")["TVALL_Sales_Qty"].sum().reset_index()
            df_plot = df_plot[df_plot["TVALL_Sales_Qty"] > 0]
            fig_pie_brand = px.pie(df_plot, names="Brand", values="TVALL_Sales_Qty", title="Marka Dağılımı (Pasta)", hole=0.3)
            st.plotly_chart(fig_pie_brand, use_container_width=True)
        with col2:
            st.subheader("Klasman Bazlı Tahmin Dağılımı")
            df_plot = filtered_output_df.groupby("Klasman")["TVALL_Sales_Qty"].sum().reset_index()
            df_plot = df_plot[df_plot["TVALL_Sales_Qty"] > 0].sort_values("TVALL_Sales_Qty", ascending=False)
            fig_bar_klasman = px.bar(df_plot, x="Klasman", y="TVALL_Sales_Qty", title="Klasman Dağılımı (Bar)")
            st.plotly_chart(fig_bar_klasman, use_container_width=True)
            
        st.subheader("Tahmin Isı Haritası (Marka x Klasman)")
        try:
            heatmap_df = filtered_output_df.pivot_table(index="Klasman", columns="Brand", values="TVALL_Sales_Qty", aggfunc="sum", fill_value=0)
            fig_heatmap = px.imshow(heatmap_df, text_auto=".0f", aspect="auto", color_continuous_scale="Blues", title="Marka ve Klasman Bazlı Tahmin Yoğunluğu")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.warning(f"Isı haritası oluşturulamadı (çok fazla/az veri): {e}")

        st.divider()
        st.subheader("Tahmin Sonuçlarını İndir")
        csv_bytes = convert_df_to_csv(filtered_output_df)
        st.download_button(
            label="Filtrelenmiş Sonuçları İndir (CSV)", 
            data=csv_bytes, 
            file_name="forecast_FW26_filtrelenmis.csv",
            mime="text/csv"
        )
        
    with tab4:
        st.header("Model Performansı ve Güvenilirlik")
        st.caption("Bu bölümde, model seçim sürecinin teknik detayları yer almaktadır.")
        
        st.subheader("Model Karşılaştırması (5-Fold Cross Validation)")
        st.dataframe(st.session_state.results_df.style.highlight_min(subset=["MAE"], color='lightgreen')
                                     .highlight_max(subset=["R2"], color='lightgreen')
                                     .format({'MAE': '{:,.0f}', 'R2': '{:.3f}'}))
        
        st.caption("MAE (Ortalama Mutlak Hata): Tahminlerin ortalama kaç adet saptığını gösterir (Düşük = İyi).")
        st.caption("R² (R-Kare): Satışlardaki değişimin ne kadarının model tarafından açıklandığını gösterir (Yüksek = İyi).")
        
        # Model Karşılaştırma Grafiği
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=st.session_state.results_df["Model"], y=st.session_state.results_df["MAE"], name="MAE (Ort. Hata)", yaxis="y", marker_color='lightblue'))
        fig_comp.add_trace(go.Scatter(x=st.session_state.results_df["Model"], y=st.session_state.results_df["R2"], name="R² (Başarı Skoru)", yaxis="y2", marker_color='darkorange'))
        fig_comp.update_layout(
            title="Model Performans Karşılaştırması (MAE vs R²)",
            yaxis=dict(title="MAE (Düşük = İyi)"),
            yaxis2=dict(title="R² (Yüksek = İyi)", overlaying="y", side="right", range=[0, 1]),
            legend=dict(x=0.1, y=1.2)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.divider()
        
        st.subheader(f"Modelin Öğrendikleri ({st.session_state.best_model_name} Özellik Önemi)")
        importance_df = get_feature_importance(st.session_state.best_model, st.session_state.X_columns)
        
        if importance_df is not None:
            fig_imp = px.bar(importance_df.sort_values("Importance", ascending=True), 
                             x="Importance", 
                             y="Feature", 
                             title="Modelin Karar Verirken Kullandığı En Önemli 15 Değişken",
                             orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(f"Seçilen model ({st.session_state.best_model_name}) özellik önemi (feature importance) desteklememektedir.")


elif page == "📈 Optimizasyon (Karar Modeli)":
    # ... (Yukarıdaki st.title ve st.stop() kontrolleri aynı kalacak) ...

    if st.session_state.opt_input_df is None:
        st.error(f"Optimizasyon girdi verisi bulunamadı.")
        st.warning("Lütfen 'Talep Tahmini' sayfasına gidin ve butona basın.")
        st.stop()
    else:
        st.info(f"Optimizasyon için verisi başarıyla yüklendi.")
    
    # Yeni Güvenli Tanımlama:
    # Veriyi kontrol ettikten sonra burada tanımlıyoruz.
    data_raw = st.session_state.opt_input_df.copy()

    with st.expander("Ham Veri Önizlemesi"): 
        st.dataframe(data_raw.head())
    
    if st.button("Optimizasyonu Başlat", type="primary", key="run_optimization"):
    # ...
        with st.spinner("Optimizasyon modeli çalışıyor... (Pyomo + glpk)"):
            try:
                data = data_raw.copy()

                # --- Adım 1: Parametreleri Hazırla ---
                index_set = data.index.tolist()
                p = data["TVALL_Sales_Qty"].to_dict() # TAHMİN
                m = data["Margin"].to_dict()
                c = data["ListPrice"].to_dict()
                st_val = data["Sell_Through"].to_dict()
                line_type = data["Line"].to_dict()
                channel = data["Channel"].to_dict()

                # --- Sidebar'dan Stratejik Parametreleri Al ---
                ANA_BUTCE = Ana_Butce_input
                OTB_PAYI_YUZDE = OTB_Payi_input
                BASIC_MIN_ORAN = basic_ratio_input[0] / 100.0
                BASIC_MAX_ORAN = basic_ratio_input[1] / 100.0
                MARGIN_MIN_ORAN = margin_min_input / 100.0
                SELL_THROUGH_CARPAN = 500
                CHANNEL_SHARES = {0: 0.6, 1: 0.4}
                
                Kullanilabilir_Butce = ANA_BUTCE * (OTB_PAYI_YUZDE / 100.0)

                # --- Adım 2: Pyomo Modelini Kur ---
                model = ConcreteModel("Showroom_Optimization")
                model.x = Var(index_set, domain=NonNegativeIntegers)
                
                def obj_rule(model): return sum(p[i] * model.x[i] for i in index_set)
                model.objective = Objective(rule=obj_rule, sense=maximize)

                def budget_rule(model): return sum(c[i] * model.x[i] for i in index_set) <= Kullanilabilir_Butce
                model.BudgetConstraint = Constraint(rule=budget_rule)

                basic_indices = [i for i in index_set if str(line_type[i]).lower() == "basic"]
                line_indices = [i for i in index_set if str(line_type[i]).lower() == "line"]
                all_indices = basic_indices + line_indices
                
                def basic_min_rule(model): return sum(model.x[i] for i in basic_indices) >= BASIC_MIN_ORAN * sum(model.x[i] for i in all_indices)
                model.BasicMinConstraint = Constraint(rule=basic_min_rule)
                def basic_max_rule(model): return sum(model.x[i] for i in basic_indices) <= BASIC_MAX_ORAN * sum(model.x[i] for i in all_indices)
                model.BasicMaxConstraint = Constraint(rule=basic_max_rule)

                def sellthrough_rule(model, i): return model.x[i] <= st_val[i] * SELL_THROUGH_CARPAN
                model.SellThroughConstraint = Constraint(index_set, rule=sellthrough_rule)

                def range_plan_rule(model, i): return model.x[i] <= p[i]
                model.RangePlanConstraint = Constraint(index_set, rule=range_plan_rule)

                total_x_expr = sum(model.x[i] for i in index_set)
                for ch, share in CHANNEL_SHARES.items():
                    indices = [i for i in index_set if channel[i] == ch]
                    model.add_component(f"ChannelShare_{ch}", Constraint(expr = sum(model.x[i] for i in indices) == share * total_x_expr))

                def avg_margin_rule(model): return sum(m[i] * model.x[i] for i in index_set) >= MARGIN_MIN_ORAN * sum(model.x[i] for i in index_set)
                model.AvgMarginConstraint = Constraint(rule=avg_margin_rule)

                # --- Adım 3: Modeli Çöz ---
                solver = SolverFactory("glpk")
                results = solver.solve(model, tee=False) 

                if (results.solver.status != 'ok') or (results.solver.termination_condition != 'optimal'):
                    st.error(f"HATA: Model optimal bir çözüm bulamadı. Durum: {results.solver.termination_condition}")
                    st.info("Kısıtları (özellikle Bütçe veya Marj) gevşetmeyi deneyin.")
                    st.stop()

                # --- Adım 4: Sonuçları İşle ---
                data["Optimal_SKU_FW26"] = [round(value(model.x[i])) for i in index_set]
                data_final = data.copy()
            
            except Exception as e:
                st.error(f"Optimizasyon sırasında bir hata oluştu: {e}")
                st.info("glpk solver'ın sisteminizde kurulu olduğundan emin olun.")
                st.stop() 

        # ====================================================
        # 🚀 YÖNETİCİ DASHBOARD'U GÖSTERİMİ
        # ====================================================
        
        st.success("✅ Optimizasyon başarıyla tamamlandı!")

        # --- KPI Hesaplamaları ---
        total_sku = data_final["Optimal_SKU_FW26"].sum()
        
        if total_sku == 0:
            st.warning("Model bir çözüm buldu ancak optimal SKU sayısı 0. Kısıtlar çok sıkı olabilir.")
            st.stop()
            
        total_budget_used = (data_final["Optimal_SKU_FW26"] * data_final["ListPrice"]).sum()
        budget_util_percent = (total_budget_used / Kullanilabilir_Butce) * 100
        avg_margin_realized = (sum(data_final["Margin"] * data_final["Optimal_SKU_FW26"]) / total_sku)
        basic_sku_sum = data_final[data_final['Line'].str.lower() == 'basic']['Optimal_SKU_FW26'].sum()
        basic_ratio_realized = (basic_sku_sum / total_sku)

        # --- 1. Yönetici Özeti (KPI Metrikleri) ---
        
        # === SON İSTEK: Başlık Değişikliği ===
        st.subheader("📈 KPI Dashboard") 
        # ==================================
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Toplam Potansiyel (Amaç)", f"{value(model.objective):,.0f}")
        col2.metric("📦 Toplam Optimal SKU", f"{total_sku:,.0f} Adet")
        col3.metric("💰 Kullanılan Bütçe", f"{total_budget_used:,.0f} TL")
        col4.metric("📊 Bütçe Kullanım Oranı", f"{budget_util_percent:.1f} %")

        st.divider()

        # --- 2. Kısıt Karnesi ---
        st.subheader("⚖️ Stratejik Kısıtların Performansı")
        
        c1, c2, c3 = st.columns(3)
        
        with c1: # Bütçe
            st.markdown(f"<h5 style='text-align: center;'>💰 Bütçe</h5>", unsafe_allow_html=True)
            st.metric("Kullanılabilir Bütçe", f"{Kullanilabilir_Butce:,.0f} TL")
            st.metric("Kullanılan Bütçe", f"{total_budget_used:,.0f} TL")

        with c2: # Marj
            st.markdown(f"<h5 style='text-align: center;'>📈 Marj</h5>", unsafe_allow_html=True)
            st.metric("Gerçekleşen Ort. Marj", f"{avg_margin_realized*100:.2f} %")
            st.metric("Hedef Min. Marj", f"{MARGIN_MIN_ORAN*100:.2f} %",
                      delta=f"{(avg_margin_realized - MARGIN_MIN_ORAN)*100:.2f} %", delta_color="normal")
                      
        with c3: # Basic Oranı
            st.markdown(f"<h5 style='text-align: center;'>🎨 Basic/Line Oranı</h5>", unsafe_allow_html=True)
            st.metric("Gerçekleşen Basic Oranı", f"{basic_ratio_realized*100:.1f} %")
            st.metric("Hedef Aralık", f"{BASIC_MIN_ORAN*100:.1f}% - {BASIC_MAX_ORAN*100:.1f}%")

        # Kanal Payı Karnesi
        st.markdown(f"<h5 style='text-align: center; margin-top: 20px;'>📺 Kanal Payları</h5>", unsafe_allow_html=True)
        cols_channel = st.columns(len(CHANNEL_SHARES))
        for idx, (ch, share) in enumerate(CHANNEL_SHARES.items()):
            channel_sum = data_final[data_final["Channel"] == ch]["Optimal_SKU_FW26"].sum()
            realized_share = (channel_sum / total_sku)
            cols_channel[idx].metric(f"Kanal {ch} Payı (Hedef {share:.0%})", 
                                     f"{realized_share:.1%}",
                                     delta=f"{(realized_share - share):.1%}", delta_color="off")
        
        st.divider()

        # --- 3. Görsel Dağılım Analizi ---
        st.subheader("📊 Dağılım Analizi (SKU Adetleri)")
        
        plot_tabs = st.tabs(["Marka'ya Göre", "Klasman'a Göre", "Line'a Göre"])
        
        with plot_tabs[0]: # Marka
            df_brand = data_final.groupby("Brand")["Optimal_SKU_FW26"].sum().reset_index()
            df_brand = df_brand[df_brand["Optimal_SKU_FW26"] > 0]
            fig_brand = px.pie(df_brand, names="Brand", values="Optimal_SKU_FW26", title="SKU Dağılımı (Marka)", hole=0.3)
            st.plotly_chart(fig_brand, use_container_width=True)

        with plot_tabs[1]: # Klasman
            df_klasman = data_final.groupby("Klasman")["Optimal_SKU_FW26"].sum().reset_index()
            df_klasman = df_klasman[df_klasman["Optimal_SKU_FW26"] > 0].sort_values("Optimal_SKU_FW26", ascending=False)
            fig_klasman = px.bar(df_klasman, x="Klasman", y="Optimal_SKU_FW26", title="SKU Dağılımı (Klasman)")
            st.plotly_chart(fig_klasman, use_container_width=True)
            
        with plot_tabs[2]: # Line
            df_line = data_final.groupby("Line")["Optimal_SKU_FW26"].sum().reset_index()
            df_line = df_line[df_line["Optimal_SKU_FW26"] > 0]
            fig_line = px.pie(df_line, names="Line", values="Optimal_SKU_FW26", title="SKU Dağılımı (Line)")
            st.plotly_chart(fig_line, use_container_width=True)

        st.divider()
        
        # --- 4. Detaylı Plan ve İndirme ---
        st.subheader("📂 Optimal Plan (Detaylı Liste)")
        
        data_to_show = data_final[data_final["Optimal_SKU_FW26"] > 0].sort_values("Optimal_SKU_FW26", ascending=False)
        st.info(f"Model, {len(data_final)} segment arasından {len(data_to_show)} segmente SKU ataması yaptı.")
        st.dataframe(data_to_show)
        
        final_csv_data = convert_df_to_csv(data_to_show)
        st.download_button(
            label="💾 Optimal Planı Excel (CSV) Olarak İndir",
            data=final_csv_data,
            file_name="optimal_showroom_plani.csv",
            mime="text/csv",
        )
