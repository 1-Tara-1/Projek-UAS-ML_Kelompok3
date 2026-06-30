import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman Utama Streamlit
st.set_page_config(
    page_title="Aplikasi Pembelajaran Mesin UAS",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk memuat model secara aman dengan caching agar aplikasi cepat
@st.cache_resource
def load_models():
    try:
        model_klasifikasi = joblib.load(
            "BestModel_Klasifikasi_RandomForest_Kelompok3.pkl"
        )

        model_regresi = joblib.load(
            "BestModel_Regresi_Ridge_Kelompok3.pkl"
        )

        return model_klasifikasi, model_regresi

    except FileNotFoundError:
        st.error(
            "File model (.pkl) tidak ditemukan. Pastikan kedua file model berada pada folder yang sama dengan app.py."
        )
        return None, None

    except Exception as e:
        st.error(
            f"Gagal memuat model. Error: {e}"
        )
        return None, None

model_klasifikasi, model_regresi = load_models()

# Navigasi Menu di Sidebar (Sesuai Aturan Integrasi Tugas UAS)
st.sidebar.title("Navigation Menu")
st.sidebar.write("### UAS Pembelajaran Mesin")
menu = st.sidebar.radio("Pilih Fitur Aplikasi:", ["Dashboard & Info", "Klasifikasi Gaya Belajar (VARK)", "Prediksi IPK Mahasiswa"])

st.sidebar.markdown("---")
st.sidebar.write("**Anggota Kelompok:**")
st.sidebar.write("1. Jonathan Immanuel Suwarno (230712367)")
st.sidebar.write("2. Qintara Wiksa Satyawira (230712380)")

# ==========================================
# MENU 1: DASHBOARD & INFO
# ==========================================
if menu == "Dashboard & Info":
    st.title("🎓 Aplikasi Analisis Akademik & Preferensi Belajar Mahasiswa")
    st.subheader("Projek Ujian Akhir Semester Genap T.A. 2025/2026")
    st.write("""
    Aplikasi ini dirancang untuk mengimplementasikan dua model Pembelajaran Mesin utama yang dikembangkan secara kokoh:
    1. **Model Klasifikasi Gaya Belajar (VARK):** Memprediksi kecenderungan gaya belajar dominan mahasiswa (Visual, Auditory, Read/Write, Kinesthetic).
    2. **Model Regresi Prediksi IPK:** Mengestimasi capaian Indeks Prestasi Kumulatif (IPK) berdasarkan parameter demografi, lingkungan sosial, dan durasi belajar mahasiswa.
    """)
    st.info("Gunakan menu navigasi di sebelah kiri untuk berpindah halaman dan mencoba melakukan prediksi input data baru.")

# ==========================================
# MENU 2: KLASIFIKASI GAYA BELAJAR
# ==========================================
elif menu == "Klasifikasi Gaya Belajar (VARK)":
    st.title("🧠 Klasifikasi Preferensi Gaya Belajar (VARK)")
    st.write("Masukkan rata-rata skor kuesioner indikator preferensi Anda berdasarkan dimensi instrumen VARK.")
    
    col1, col2 = st.columns(2)
    with col1:
        score_visual = st.slider("Rata-rata Skor Dimensi Visual (C1):", 1.0, 5.0, 3.5, 0.1)
        score_auditory = st.slider("Rata-rata Skor Dimensi Auditory (C2):", 1.0, 5.0, 3.5, 0.1)
    with col2:
        score_readwrite = st.slider("Rata-rata Skor Dimensi Read/Write (C3):", 1.0, 5.0, 3.5, 0.1)
        score_kinesthetic = st.slider("Rata-rata Skor Dimensi Kinesthetic (C4):", 1.0, 5.0, 3.5, 0.1)
        
    st.markdown("##### Indikator Demografi / Faktor Pendukung:")
    col3, col4 = st.columns(2)
    with col3:
        program_studi = st.selectbox("Program Studi:", ["Informatika", "Sistem Informasi", "Teknik Industri"])
        durasi_belajar = st.number_input("Rata-rata Jam Belajar Mandiri per Hari:", 0.0, 24.0, 2.5, 0.5)
    with col4:
        kehadiran_kelas = st.slider("Persentase Kehadiran Kuliah (%):", 0, 100, 90, 5)

    if st.button("Klasifikasikan Gaya Belajar", type="primary"):
        if model_klasifikasi is not None:
            # Sesuaikan urutan dan nama kolom dataframe input dengan X_train pada 'Untitled-1-revisi.ipynb' Anda
            input_data = pd.DataFrame([{
                'Visual_Mean': score_visual,
                'Auditory_Mean': score_auditory,
                'ReadWrite_Mean': score_readwrite,
                'Kinesthetic_Mean': score_kinesthetic,
                'Program_Studi': program_studi,
                'Durasi_Belajar': durasi_belajar,
                'Kehadiran': kehadiran_kelas
                # TAMBAHKAN KOLOM LAIN APABILA MODEL ANDA MENGGUNAKAN FITUR TAMBAHAN DI NOTBOOK 1
            }])
            
            try:
                hasil_prediksi = model_klasifikasi.predict(input_data)[0]
                st.success(f"### Hasil Prediksi Gaya Belajar: **{hasil_prediksi}**")
                
                # Memberikan insight edukasi singkat berdasarkan output
                if "Visual" in str(hasil_prediksi):
                    st.info("💡 **Rekomendasi:** Gunakan visualisasi peta konsep (mind mapping), grafik berwarna, atau infografis saat merangkum materi kuliah.")
                elif "Auditory" in str(hasil_prediksi):
                    st.info("💡 **Rekomendasi:** Sangat disarankan belajar melalui mendengarkan rekaman penjelasan, diskusi kelompok aktif, atau mengulang materi secara lisan.")
                elif "Read" in str(hasil_prediksi):
                    st.info("💡 **Rekomendasi:** Tulis ulang poin kuliah, rangkum jurnal/modul, dan perbanyak membaca literatur tertulis.")
                elif "Kinesthetic" in str(hasil_prediksi):
                    st.info("💡 **Rekomendasi:** Pelajari studi kasus nyata, latihan simulasi proyek, atau lakukan praktik langsung (hands-on coding).")
            except Exception as ex:
                st.error(f"Terjadi ketidaksesuaian dimensi fitur input dengan model. Silakan periksa kolom X_train Anda. Error: {ex}")
        else:
            st.warning("Model Klasifikasi tidak tersedia.")

# ==========================================
# MENU 3: PREDIKSI IPK (REGRESI)
# ==========================================
elif menu == "Prediksi IPK Mahasiswa":
    st.title("📈 Regresi Prediksi Nilai IPK Mahasiswa")
    st.write("Isi data indikator demografi dan kebiasaan akademik mahasiswa di bawah ini untuk mengestimasi raihan IPK.")

    col1, col2 = st.columns(2)
    with col1:
        jam_belajar = st.number_input("Jam Belajar per Minggu (Akademik):", 0.0, 168.0, 15.0, 1.0)
        kehadiran_persen = st.slider("Persentase Kehadiran Tatap Muka (%):", 0.0, 100.0, 95.0, 1.0)
        partisipasi_organisasi = st.radio("Aktif Organisasi/UKM?", ["Ya", "Tidak"])
    with col2:
        gaya_belajar_mhs = st.selectbox("Gaya Belajar Utama:", ["Visual", "Auditory", "ReadWrite", "Kinesthetic"])
        skor_motivasi = st.slider("Skor Motivasi Diri Internal (1-10):", 1, 10, 8)
        ketersediaan_internet = st.radio("Akses Internet di Rumah Memadai?", ["Ya", "Tidak"])

    if st.button("Prediksi Nilai IPK", type="primary"):
        if model_regresi is not None:
            # Sesuaikan urutan dan nama kolom dataframe input dengan X_train pada 'prediksi ipk v2.ipynb' Anda
# Skenario input di app.py
            input_regresi = pd.DataFrame([{
                'Gender': gender,
                'UKT': ukt,
                'Gaya_Belajar_Mhs': gaya_belajar_mhs,
                'Skor_Motivasi': skor_motivasi,
                'Akses_Internet': ketersediaan_internet
            }])
            
            # --- PROSES ENCODING MANUAL (SUBSTITUSI COLUMN TRANSFORMER) ---
            # Sesuaikan nilai string dengan dataset asli Anda (Contoh di bawah ini):
            input_regresi['Gender'] = input_regresi['Gender'].map({'Pria': 1, 'Wanita': 0, 'Laki-laki': 1})
            input_regresi['Akses_Internet'] = input_regresi['Akses_Internet'].map({'Ya': 1, 'Tidak': 0})
            input_regresi['Gaya_Belajar_Mhs'] = input_regresi['Gaya_Belajar_Mhs'].map({'Visual': 0, 'Auditory': 1, 'Read/Write': 2, 'Kinesthetic': 3})
            
            # Jika UKT Anda bertipe kategori/string di dataset, lakukan mapping juga di sini.
            # Jika UKT sudah berupa angka (nominal), biarkan saja atau hapus baris di bawah ini:
            # input_regresi['UKT'] = pd.to_numeric(input_regresi['UKT']) 
            
            try:
                # Sekarang model_regresi hanya menerima angka murni tanpa terikat versi pipeline scikit-learn
                ipk_prediksi = model_regresi.predict(input_regresi)[0]
                
                # Batasi keluaran IPK agar rasional sesuai standar akademik (0.00 - 4.00)
                ipk_final = max(0.0, min(4.0, float(ipk_prediksi)))
                
                st.success(f"### Estimasi Raihan IPK Mahasiswa: **{ipk_final:.2f}**")
                
                # Batasi keluaran IPK agar rasional sesuai standar akademik (0.00 - 4.00)
                ipk_final = max(0.0, min(4.0, float(ipk_prediksi)))
                
                st.success(f"### Estimasi Raihan IPK Mahasiswa: **{ipk_final:.2f}**")
                
                # Berikan indikator visual performa akademik
                if ipk_final >= 3.51:
                    st.balloons()
                    st.markdown("🏅 **Predikat Prediksi:** *Dengan Pujian (Cum Laude)*. Pertahankan ritme belajar Anda!")
                elif ipk_final >= 3.00:
                    st.markdown("👍 **Predikat Prediksi:** *Sangat Memuaskan*. Model menunjukkan kompetensi akademik yang solid.")
                else:
                    st.markdown("⚠️ **Catatan Evaluasi:** Model mendeteksi risiko penurunan capaian. Perlu adanya optimalisasi durasi belajar mandiri.")
            except Exception as ex:
                st.error(f"Terjadi ketidaksesuaian dimensi fitur input dengan model regresi. Periksa pipeline preprocessing notebook 2 Anda. Error: {ex}")
        else:
            st.warning("Model Regresi tidak tersedia.")