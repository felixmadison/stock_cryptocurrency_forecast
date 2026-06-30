import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import warnings
from tensorflow.keras.models import load_model
from datetime import datetime


# Menyembunyikan peringatan versi library agar dashboard bersih
warnings.filterwarnings("ignore")

@st.cache_data(ttl=3600) # Simpan data kurs selama 1 jam
def get_live_exchange_rate():
    try:
        data_kurs = yf.download("IDR=X", period="1d", interval="1m")
        if not data_kurs.empty:
            return float(data_kurs['Close'].iloc[-1].item())
        return 16000.0 
    except:
        return 16000.0

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Saham & Kripto", layout="centered")

st.title("📈 Dashboard Prediksi Aset")
st.write("Aplikasi Prediksi Harga Saham dan Bitcoin Menggunakan Model LSTM")

# --- AREA INPUT (HALAMAN UTAMA) ---
st.info("Pilih konfigurasi aset di bawah ini:")
col_input1, col_input2 = st.columns(2)

with col_input1:
    pilihan_tampilkan = ["BTC", "BBCA", "BBRI", "BBNI"]
    asset = st.selectbox("Pilih Aset:", pilihan_tampilkan)

with col_input2:
    pilihan_hari = [1, 3, 5, 7]
    forecast_days = st.selectbox("Jumlah Hari Prediksi ke Depan:", pilihan_hari)

# --- LOGIKA UTAMA ---
if st.button("Jalankan Prediksi & Uji Akurasi", use_container_width=True, type="primary"):
    try:
        # 1. Load Model & Scaler
        model_path = f"models/{asset}_Total_LSTM.h5"
        scaler_path = f"models/{asset}_Total_scaler.pkl"
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        # 2. Ambil Data (100 hari agar window 60 hari aman)
        ticker_symbol = f"{asset}.JK" if asset != "BTC" else "BTC-USD"
        with st.spinner(f"Menganalisis data terbaru {asset}..."):
            df = yf.download(ticker_symbol, period="100d", interval="1d")
            
        if df.empty:
            st.error("Gagal mengambil data.")
        else:
            # --- TAHAP 1: UJI AKURASI HARI INI ---
            st.subheader("Uji Akurasi Model")
            st.write("Model mencoba menebak harga **HARI INI** tanpa melihat data aslinya.")

            # Harga asli hari ini untuk pembanding
            harga_asli_hari_ini = float(df['Close'].iloc[-1].item())
            
            # Ambil data 60 hari SEBELUM hari ini (indeks -61 sampai -1)
            # Ini memastikan model "buta" terhadap harga hari ini
            data_uji = df['Close'].iloc[-61:-1].values.reshape(-1, 1)
            data_uji_scaled = scaler.transform(data_uji)
            input_uji = data_uji_scaled.reshape(1, 60, 1)
            
            # Prediksi hari ini
            pred_uji_scaled = model.predict(input_uji, verbose=0)
            harga_tebakan_model = float(scaler.inverse_transform(pred_uji_scaled)[0][0])

            # Hitung Selisih & Akurasi
            selisih_uji = abs(harga_tebakan_model - harga_asli_hari_ini)
            persentase_akurasi = (1 - (selisih_uji / harga_asli_hari_ini)) * 100

            # Tampilan Metrik Akurasi
            sym = "$" if asset == "BTC" else "Rp"
            # Menampilkan 2 desimal untuk semua aset agar perubahan intraday sekecil apapun terlihat
            fmt = "{:,.2f}"
            
            ua1, ua2,  = st.columns(2)
            ua1.metric("Harga Asli", f"{sym} {fmt.format(harga_asli_hari_ini)}")
            ua2.metric("Tebakan Model", f"{sym} {fmt.format(harga_tebakan_model)}")
            ##ua3.metric("Akurasi", f"{persentase_akurasi:.2f}%")
            
            ##st.progress(persentase_akurasi / 100)
            st.divider()

            # --- TAHAP 2: PREDIKSI MASA DEPAN ---
            st.subheader(f"Prediksi {forecast_days} Hari ke Depan")
            st.caption(f"Waktu Kalkulasi: {datetime.now().strftime('%d %B %Y - %H:%M:%S')}")
            
            # Sekarang baru pakai data sampai hari ini untuk tebak masa depan
            last_60_days = df['Close'].tail(60).values.reshape(-1, 1)
            curr_step = scaler.transform(last_60_days).reshape(1, 60, 1)
            
            for _ in range(forecast_days):
                pred_scaled = model.predict(curr_step, verbose=0)
                new_step = np.append(curr_step[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
                curr_step = new_step
                
            harga_prediksi_depan = float(scaler.inverse_transform(pred_scaled)[0][0])

            # Tampilkan Hasil Prediksi Masa Depan
            selisih_depan = harga_prediksi_depan - harga_asli_hari_ini
            persen_depan = (selisih_depan / harga_asli_hari_ini) * 100

            if asset == "BTC":
                kurs = get_live_exchange_rate()
                st.info(f"Kurs Live: **1 USD = Rp {kurs:,.2f}**")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Harga Saat Ini", f"$ {harga_asli_hari_ini:,.2f}")
                    st.caption(f"Estimasi: Rp {harga_asli_hari_ini * kurs:,.0f}")
                with c2:
                    st.metric(f"Prediksi {forecast_days} Hari", f"$ {harga_prediksi_depan:,.2f}", f"{persen_depan:.2f}%")
                    st.caption(f"Estimasi: Rp {harga_prediksi_depan * kurs:,.0f}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Harga Saat Ini", f"Rp {harga_asli_hari_ini:,.2f}")
                c2.metric(f"Prediksi {forecast_days} Hari", f"Rp {harga_prediksi_depan:,.2f}", f"{persen_depan:.2f}%")


            # --- 6. VISUALISASI GRAFIK ---
            st.write("### Visualisasi Tren")
            df_hist = df['Close'].tail(30).copy()
            last_date = df.index[-1]
            future_date = last_date + pd.Timedelta(days=forecast_days)
            
            chart_data = pd.DataFrame(df_hist)
            chart_data.columns = ['Harga Historis']
            
            # Garis proyeksi menyambung dari harga asli hari ini ke prediksi masa depan
            df_proj = pd.DataFrame(
                {'Proyeksi': [harga_asli_hari_ini, harga_prediksi_depan]}, 
                index=[last_date, future_date]
            )
            
            final_chart = pd.concat([chart_data, df_proj])
            st.line_chart(final_chart, color=["#1f77b4", "#ff7f0e"])

            # Kesimpulan
            if selisih_depan >= 0:
                st.success(f"Analisis: Aset **{asset}** diprediksi Naik dalam {forecast_days} hari.")
            else:
                st.warning(f"Analisis: Aset **{asset}** diprediksi Turun dalam {forecast_days} hari.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.divider()
st.caption("Penelitian Tugas Akhir - Uji Validasi Model LSTM")
st.caption("⚠️ Disclaimer: Prediksi AI tidak menjamin akurasi 100%.")