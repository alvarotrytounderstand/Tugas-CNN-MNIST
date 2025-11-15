# -----------------------------------------------------------------
# NAMA FILE: app.py
# (KODE YANG DIPERBARUI - Layout di Tengah)
# -----------------------------------------------------------------
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Tugas 4: Klasifikasi Angka CNN", layout="wide")
st.markdown("<h1 style='text-align: center;'>Tugas 4: Klasifikasi Angka (CNN)</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Gambar satu angka (0-9) di kotak hitam di bawah ini.</h3>", unsafe_allow_html=True)


# --- 1. Memuat Model ---
MODEL_PATH = 'mnist_cnn_model.h5'

@st.cache_resource
def load_keras_model():
    """
    Memuat model .h5 yang sudah dilatih dari file lokal.
    """
    try:
        model = load_model(MODEL_PATH)
        print("Model CNN MNIST berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"Error saat memuat model '{MODEL_PATH}': {e}")
        st.error("Pastikan file 'mnist_cnn_model.h5' ada di folder yang sama dengan 'app.py' di GitHub.")
        return None

model = load_keras_model()

if model:
    # --- 2. Membuat Papan Gambar (Canvas) ---
    drawing_size = 336 # Ukuran 280x280 (kita buat 12x lipat dari 28x28)
    
    # PERBAIKAN: Gunakan kolom untuk menengahkan canvas
    col1, col2, col3 = st.columns([1, 1, 1]) # [kosong, isi, kosong]

    with col2: # Semua widget di dalam kolom tengah
        # st.subheader("Gambar Angka Anda di Sini:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # Latar belakang transparan
            stroke_width=25, # Ukuran kuas (dibuat tebal agar mirip data MNIST)
            stroke_color="#FFFFFF", # Warna kuas putih
            background_color="#000000", # Latar belakang hitam
            width=drawing_size,
            height=drawing_size,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True # Pastikan gambar terkirim saat tombol ditekan
        )

        # PERBAIKAN: Pindahkan tombol ke kolom tengah
        run_button = st.button("Prediksi Angka", use_container_width=True, type="primary")

    st.markdown("---")

    # --- 3. Logika Preprocessing dan Prediksi ---
    # PERBAIKAN: Pindahkan hasil prediksi ke bawah dan ke tengah
    
    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])

    if run_button: # Jika tombol ditekan
        with col_res2: # Tampilkan hasil di kolom tengah
            if canvas_result.image_data is not None:
                st.subheader("Hasil Prediksi:")
                try:
                    # Ambil gambar dari canvas (format RGBA)
                    img_rgba = canvas_result.image_data
                    
                    # Ubah ke Grayscale (hitam putih)
                    img_gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
                    
                    # Resize gambar dari 336x336 ke 28x28 (ukuran input model MNIST)
                    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
                    
                    # Ubah format data agar sesuai dengan input TensorFlow
                    # (1, 28, 28, 1) -> 1 gambar, ukuran 28x28, 1 channel (grayscale)
                    img_input = img_resized.reshape(1, 28, 28, 1)
                    
                    # Normalisasi data (dari 0-255 menjadi 0-1)
                    img_input = img_input.astype('float32') / 255.0

                    # Lakukan prediksi
                    with st.spinner("Model sedang menebak..."):
                        prediction = model.predict(img_input)
                        predicted_number = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                    
                    st.success(f"Prediksi Selesai!")
                    
                    # Tampilkan hasil
                    st.markdown(f"<h2 style='text-align: center; color: #28a745;'>Model menebak: {predicted_number}</h2>", unsafe_allow_html=True)
                    st.write(f"Tingkat keyakinan: **{confidence:.2f}%**")
                    
                    # Tampilkan gambar yang dilihat model (setelah di-resize)
                    with st.expander("Lihat gambar input 28x28 (yang dilihat model)"):
                        st.image(img_resized, width=140, caption="Input 28x28")

                except Exception as e:
                    st.error(f"Terjadi error saat preprocessing gambar: {e}")
            else:
                st.warning("Silakan gambar sesuatu di canvas terlebih dahulu.")
else:
    st.error("Aplikasi tidak dapat berjalan karena model 'mnist_cnn_model.h5' gagal dimuat.")
