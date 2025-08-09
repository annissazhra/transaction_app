# ============================================================
# 💰 APLIKASI PREDIKSI JUMLAH TRANSAKSI - DECISION TREE
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import os

# ============================================================
# 1️⃣ Load Model Pipeline
# ============================================================
@st.cache_resource # Gunakan cache_resource untuk memuat aset hanya sekali saat aplikasi berjalan
def load_full_pipeline():
    """
    Memuat pipeline lengkap (preprocessor dan model) yang sudah terlatih.
    """
    pipeline_path = "model_transaction_pipeline.pkl"

    if not os.path.exists(pipeline_path):
        st.error(f"❌ File pipeline '{pipeline_path}' tidak ditemukan.")
        st.error("Pastikan file ini sudah diunggah ke repositori GitHub Anda di direktori yang sama.")
        st.stop() # Hentikan eksekusi aplikasi jika file tidak ada

    try:
        # Memuat pipeline yang sudah dilatih
        full_pipeline = joblib.load(pipeline_path)
        return full_pipeline
    except Exception as e:
        st.error(f"❌ Gagal memuat pipeline: {e}.")
        st.error("Pastikan versi pustaka Anda konsisten dengan saat model dilatih.")
        st.stop()

# Muat pipeline saat aplikasi dimulai
loaded_pipeline = load_full_pipeline()

# ============================================================
# 2️⃣ Konfigurasi Aplikasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Prediksi Jumlah Transaksi",
    page_icon="💰",
    layout="centered" # Layout centered agar terlihat bagus di berbagai ukuran layar
)

st.title("💰 Aplikasi Prediksi Jumlah Transaksi (Decision Tree)")
st.markdown("Masukkan detail transaksi untuk memprediksi jumlahnya.")

# ============================================================
# 3️⃣ Formulir Input Pengguna
# ============================================================
with st.form("transaction_amount_form"):
    st.subheader("📝 Masukkan Detail Transaksi")

    # Input untuk fitur kategorikal
    status_options = ['Success', 'Failed', 'Pending', 'Cancelled']
    status = st.selectbox("Status Transaksi", status_options)

    sender_upi_id = st.text_input("Sender UPI ID (contoh: user123@upi)")
    receiver_upi_id = st.text_input("Receiver UPI ID (contoh: merchantabc@upi)")

    # Jika ada fitur numerik lain yang tidak dibuang dan perlu diinput
    # Contoh:
    # numeric_feature_example = st.number_input("Fitur Numerik Contoh", min_value=0, value=100)

    submitted = st.form_submit_button("🔮 Prediksi Jumlah")

    if submitted:
        try:
            # Buat DataFrame dari input pengguna
            # PASTIKAN NAMA KOLOM DAN URUTANNYA SESUAI DENGAN DATA PELATIHAN ASLI
            input_df = pd.DataFrame([{
                'Status': status,
                'Sender UPI ID': sender_upi_id,
                'Receiver UPI ID': receiver_upi_id
                # Tambahkan kolom numerik lain di sini jika ada
                # 'Nama_Fitur_Numerik': numeric_feature_example,
            }])

            # Melakukan prediksi menggunakan pipeline lengkap
            prediction = loaded_pipeline.predict(input_df)

            st.subheader("✅ Prediksi Berhasil!")
            st.success(f"Jumlah transaksi yang diprediksi adalah: ₹{prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
            st.info("Silakan periksa input Anda atau coba lagi nanti.")
            st.exception(e) # Menampilkan detail error untuk debugging