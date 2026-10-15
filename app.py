import math
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman Utama Streamlit
st.set_page_config(
    page_title="Cosinus Similarity",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Penghitungan  Kemiripan dengan Cosinus")
st.write(
    "Aplikasi web interaktif untuk menganalisis kemiripan kalimat/dokumen."
)

st.markdown("---")

# Section 1: Form Input Kalimat (1 sampai 5)
st.subheader("1. Masukkan Kalimat")
st.caption(
    "Isi minimal 2 kalimat untuk dibandingkan. Kalimat yang kosong akan diabaikan secara otomatis."
)

sentences_input = []
for i in range(1, 6):
    text = st.text_area(
        f"Kalimat / Dokumen {i}:",
        key=f"sentence_{i}",
        placeholder=f"Ketik kalimat {i} di sini...",
        height=70,
    )
    if text.strip():
        sentences_input.append((f"Dokumen {i}", text.strip()))

st.markdown("---")

# Section 2: Penentuan Atribut (Otomatis vs Kustom)
st.subheader("2. Mode Penentuan Atribut (Kata Kunci)")

mode_atribut = st.radio(
    "Pilih bagaimana atribut/fitur kata kunci ditentukan:",
    options=[
        "🤖 Otomatis (Mengekstrak seluruh kata penting dari kalimat)",
        "✍️ Manual / Sesuai Kehendak (Ditentukan sendiri kata kuncinya)",
    ],
)

custom_keywords_input = ""
if "Manual" in mode_atribut:
    custom_keywords_input = st.text_input(
        "Masukkan kata kunci yang diinginkan (pisahkan dengan koma):",
        placeholder="Ketik kata kunci disini...",
        help="Aplikasi hanya akan menghitung frekuensi berdasarkan kata-kata yang kamu ketik di sini.",
    )

# Tombol Proses
process_btn = st.button("🚀 Proses Penghitungan & Buat Grafik", type="primary")

# Section 3: Pemrosesan & Hasil
if process_btn:
    if len(sentences_input) < 2:
        st.error("⚠️ Silakan masukkan minimal **2 kalimat** untuk dibandingkan!")
    elif "Manual" in mode_atribut and not custom_keywords_input.strip():
        st.error("⚠️ Kamu memilih Mode Manual. Silakan isi kata kuncinya terlebih dahulu!")
    else:
        labels = [item[0] for item in sentences_input]
        docs = [item[1] for item in sentences_input]

        # Preprocessing Sederhana: Lowercase & Hapus Tanda Baca
        cleaned_docs = [re.sub(r'[^\w\s]', ' ', doc.lower()) for doc in docs]

        if "Manual" in mode_atribut:
            # Mode Manual: Pakai atribut yang diketik user
            features = [k.strip().lower() for k in custom_keywords_input.split(",") if k.strip()]
            
            # Matriks Frekuensi Manual
            X_list = []
            for doc in cleaned_docs:
                row = []
                for feat in features:
                    count = len(re.findall(r'\b' + re.escape(feat) + r'\b', doc))
                    row.append(count)
                X_list.append(row)
            X = np.array(X_list)
            
        else:
            # Mode Otomatis: Pakai CountVectorizer dengan Stopwords Bahasa Indonesia
            stopwords_id = [
                # Kata Hubung (Konjungsi)
                "dan", "atau", "serta", "sambil", "tetapi", "melainkan", "sedangkan", "namun", 
                "sebab", "karena", "karenanya", "sehingga", "maka", "sampai", "hingga", "agar", 
                "supaya", "jika", "jikalau", "kalau", "apabila", "bila", "asalkan", "meskipun", 
                "walaupun", "biarpun", "kendati", "seakan", "seolah", "seperti", "sebagai", 
                "sebagaimana", "bahkan", "bahkanpun", "maupun", "lalu", "kemudian", "bahwa", 
                "sebelum", "sesudah", "setelah", "sewaktu", "ketika", "saat", "sementara", 
            
                # Kata Depan (Preposisi)
                "di", "ke", "dari", "pada", "dalam", "dengan", "oleh", "tentang", "mengenai", 
                "terhadap", "bersama", "beserta", "tanpa", "bagi", "untuk", "guna", "demi", 
                "antara", "melalui", "secara",
            
                # Kata Ganti (Pronomina)
                "ini", "itu", "tersebut", "berikut", "demikian", "sini", "situ", "sana", 
                "saya", "aku", "kami", "kita", "kamu", "engkau", "kalian", "dia", "ia", 
                "mereka", "anda", "siapa", "apa", "mana", "yang",
            
                # Kata Keterangan, Tunjuk & Penegas
                "ada", "adalah", "ialah", "merupakan", "yaitu", "yakni", "akan", "telah", 
                "sudah", "sedang", "masih", "belum", "pernah", "bisa", "dapat", "boleh", 
                "harus", "mesti", "perlu", "sangat", "amat", "paling", "terlalu", "agak", 
                "kurang", "lebih", "pula", "pun", "juga", "saja", "lagi", "hanya", "cuma", 
                "bahkan", "tentu", "memang", "pasti", "bukan", "tidak", "tanpa", "satu", 
                "dua", "tiga", "banyak", "beberapa", "sebagian", "seluruh", "semua"
            ]
            vectorizer = CountVectorizer(stop_words=stopwords_id)
            X = vectorizer.fit_transform(cleaned_docs).toarray()
            features = vectorizer.get_feature_names_out()

        st.markdown("---")
        st.header("📊 Hasil Penghitungan Lengkap")

        # LANDASAN RUMUS MATHEMATIS
        with st.expander("ℹ️ **Lihat Rumus & Landasan Perhitungan Matematika**", expanded=True):
            st.latex(r"\text{sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}")
            st.latex(r"\text{Sudut Derajat } (\theta) = \arccos(\text{sim}(x, y)) \times \frac{180^\circ}{\pi}")
            
            st.markdown("""
            **Langkah-Langkah Perhitungan:**
            1. **Perkalian Titik ($x \cdot y$) :**
               $$x \cdot y = (x_1 \cdot y_1) + (x_2 \cdot y_2) + \dots + (x_n \cdot y_n)$$
            2. **Panjang Vektor ($\|x\|$ dan $\|y\|$):**
               $$\|x\| = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$$
               dan $$\|y\| = \sqrt{y_1^2 + y_2^2 + \dots + y_n^2}$$
            """)

        # Matriks Frekuensi Kata
        st.subheader("a. Matriks Frekuensi Kata (Atribut / Fitur)")
        st.write(f"Total **{len(features)} atribut** yang dievaluasi:")
        df_tf = pd.DataFrame(X, index=labels, columns=features)
        st.dataframe(df_tf, use_container_width=True)

        # Hitung Cosine Similarity & Sudut Derajat
        cos_sim_matrix = cosine_similarity(X)
        num_docs = len(docs)

        angle_matrix = np.zeros((num_docs, num_docs))

        for i in range(num_docs):
            for j in range(num_docs):
                val = np.clip(cos_sim_matrix[i][j], -1.0, 1.0)
                angle_matrix[i][j] = round(math.degrees(math.acos(val)), 2)

        df_sim = pd.DataFrame(np.round(cos_sim_matrix, 4), index=labels, columns=labels)
        df_angle = pd.DataFrame(angle_matrix, index=labels, columns=labels)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("b. Matriks Cosinus Similarity")
            st.dataframe(df_sim.style.highlight_max(axis=0), use_container_width=True)
        with col2:
            st.subheader("c. Matriks Sudut Vektor (° Derajat)")
            st.dataframe(df_angle.style.highlight_min(axis=0), use_container_width=True)

        # Detail Per Pasangan
        st.subheader("d. Rincian Kemiripan & Sudut Antar Pasangan Dokumen")
        rincian_data = []
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                sim_val = cos_sim_matrix[i][j]
                deg_val = angle_matrix[i][j]
                
                # Pengkategorian Status Kemiripan
                if sim_val >= 0.8:
                    status = "Sangat Mirip (Identik)"
                elif sim_val >= 0.6:
                    status = "Mirip (Topik Dekat)"
                elif sim_val >= 0.4:
                    status = "Cukup / Agak Mirip"
                elif sim_val >= 0.2:
                    status = "Kurang Mirip"
                else:
                    status = "Sangat Berbeda"

                rincian_data.append({
                    "Pasangan": f"{labels[i]} ↔ {labels[j]}",
                    "Cosine Similarity": f"{sim_val:.4f}",
                    "Sudut (° Derajat)": f"{deg_val}°",
                    "Status Kemiripan": status,
                })

        df_rincian = pd.DataFrame(rincian_data)
        st.table(df_rincian)

        # Tombol Download Hasil ke CSV
        csv_data = df_sim.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Matriks Cosine Similarity (CSV)",
            data=csv_data,
            file_name="hasil_cosine_similarity.csv",
            mime="text/csv",
        )

        # Visualisasi Vektor 2D
        st.subheader("e. Visualisasi Vektor & Titik Koordinat")

        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
        else:
            coords = X

        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i in range(num_docs):
            x_val = coords[i, 0]
            y_val = coords[i, 1]

            fig.add_trace(
                go.Scatter(
                    x=[0, x_val],
                    y=[0, y_val],
                    mode="lines+markers+text",
                    name=labels[i],
                    text=["", labels[i]],
                    textposition="top center",
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=[0, 10]),
                )
            )

        fig.update_layout(
            title="Sistem Koordinat Vektor Kalimat (Arah Sudut Memvisualisasikan Kemiripan)",
            xaxis_title="Sumbu X (Komponen Vektor 1)",
            yaxis_title="Sumbu Y (Komponen Vektor 2)",
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
            height=600,
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "💡 **Cara Membaca Grafik:** Semakin **sempit sudut** antara dua garis vektor (mendekati 0°), "
            "artinya kedua kalimat **semakin mirip**!"
        )
