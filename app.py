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
    page_title="Pengukuran Kemiripan dengan Cosinus",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Kalkulator Cosine Similarity & Visualisasi Sudut Vektor")
st.write(
    "Aplikasi untuk menghitung tingkat kemiripan kalimat, sudut antar vektor, dan visualisasi grafik."
)

st.markdown("---")

# Section 1: Form Input Kalimat (1 sampai 5)
st.subheader("1. Masukkan Kalimat / Postingan Media Sosial")
st.caption(
    "Isi minimal 2 kalimat untuk dibandingkan. Kalimat yang kosong akan diabaikan secara otomatis."
)

sentences_input = []
for i in range(1, 6):
    text = st.text_area(
        f"Kalimat / Dokumen {i}:",
        key=f"sentence_{i}",
        placeholder=f"Ketik atau paste isi postingan {i} di sini...",
        height=70,
    )
    if text.strip():
        sentences_input.append((f"Dokumen {i}", text.strip()))

# Fitur Opsional: Penggabungan Kata Kunci Khusus
st.caption("⚙️ **Pengaturan Atribut Khusus (Frasa):**")
custom_phrases = st.text_input(
    "Frasa khusus yang digabung jadi 1 atribut (pisahkan dengan koma):",
    value="paruh waktu, penuh waktu, p3k, dpr ri, banggar dpr, pemerintah pusat",
)

# Tombol Proses
process_btn = st.button("🚀 Proses Penghitungan & Buat Grafik", type="primary")

# Section 2: Pemrosesan & Hasil
if process_btn:
    if len(sentences_input) < 2:
        st.error("⚠️ Silakan masukkan minimal **2 kalimat** untuk dibandingkan!")
    else:
        labels = [item[0] for item in sentences_input]
        docs = [item[1] for item in sentences_input]

        # Preprocessing: Normalisasi Frasa Khusus
        phrase_list = [p.strip().lower() for p in custom_phrases.split(",") if p.strip()]
        
        processed_docs = []
        for doc in docs:
            doc_clean = doc.lower()
            # Ganti spasi pada frasa khusus dengan underscore (misal: "paruh waktu" -> "paruh_waktu")
            for phrase in phrase_list:
                phrase_clean = phrase.lower()
                replacement = phrase_clean.replace(" ", "_")
                doc_clean = re.sub(r'\b' + re.escape(phrase_clean) + r'\b', replacement, doc_clean)
            processed_docs.append(doc_clean)

        # Daftar Stopwords Bahasa Indonesia untuk Diabaikan
        stopwords_id = [
            "dan", "di", "ke", "dari", "yang", "ini", "itu", "untuk", "pada", "dengan",
            "adalah", "sebagai", "akan", "juga", "oleh", "atau", "secara", "bisa", "sudah",
            "ada", "bahwa", "satu", "dapat", "tersebut", "ini", "itu", "karena", "maka",
            "sangat", "lagi", "pun", "kita", "kami", "mereka", "ia", "dia", "saat", "mencapai"
        ]

        # 1. Ekstraksi Kata menjadi Atribut (Bag of Words)
        vectorizer = CountVectorizer(stop_words=stopwords_id)
        X = vectorizer.fit_transform(processed_docs).toarray()
        
        # Kembalikan nama fitur yang ada underscore menjadi spasi
        raw_features = vectorizer.get_feature_names_out()
        features = [f.replace("_", " ") for f in raw_features]

        st.markdown("---")
        st.header("📊 Hasil Penghitungan Lengkap")

        # Display Matriks Frekuensi Kata
        st.subheader("a. Matriks Frekuensi Kata (Atribut / Fitur)")
        st.write(f"Total **{len(features)} atribut/kata kunci unik** berhasil diekstrak:")
        df_tf = pd.DataFrame(X, index=labels, columns=features)
        st.dataframe(df_tf, use_container_width=True)

        # 2. Hitung Cosine Similarity & Derajat Sudut
        cos_sim_matrix = cosine_similarity(X)
        num_docs = len(docs)

        # Hitung Matriks Sudut (° Derajat) & Dissimilarity
        angle_matrix = np.zeros((num_docs, num_docs))
        dissim_matrix = np.zeros((num_docs, num_docs))

        for i in range(num_docs):
            for j in range(num_docs):
                val = np.clip(cos_sim_matrix[i][j], -1.0, 1.0)
                angle_matrix[i][j] = round(math.degrees(math.acos(val)), 2)
                dissim_matrix[i][j] = round(1.0 - val, 4)

        df_sim = pd.DataFrame(np.round(cos_sim_matrix, 4), index=labels, columns=labels)
        df_angle = pd.DataFrame(angle_matrix, index=labels, columns=labels)
        df_dissim = pd.DataFrame(dissim_matrix, index=labels, columns=labels)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("b. Matriks Cosine Similarity (0 - 1)")
            st.dataframe(df_sim.style.highlight_max(axis=0), use_container_width=True)
        with col2:
            st.subheader("c. Matriks Sudut Vektor (° Derajat)")
            st.dataframe(df_angle.style.highlight_min(axis=0), use_container_width=True)

        st.subheader("d. Matriks Ketidakmiripan (Dissimilarity Matrix)")
        st.caption("Nilai mendekati 0 menunjukkan dokumen semakin identik.")
        st.dataframe(df_dissim, use_container_width=True)

        # Detail Per Pasangan
        st.subheader("e. Rincian Kemiripan Antar Pasangan Dokumen")
        rincian_data = []
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                sim_val = cos_sim_matrix[i][j]
                deg_val = angle_matrix[i][j]
                dissim_val = dissim_matrix[i][j]
                rincian_data.append({
                    "Pasangan": f"{labels[i]} ↔ {labels[j]}",
                    "Cosine Similarity": f"{sim_val:.4f}",
                    "Sudut (° Derajat)": f"{deg_val}°",
                    "Dissimilarity (1 - sim)": f"{dissim_val:.4f}",
                    "Status Kemiripan": (
                        "Sangat Mirip (Satu Topik)"
                        if sim_val > 0.7
                        else "Agak Mirip"
                        if sim_val > 0.3
                        else "Sangat Berbeda"
                    ),
                })
        st.table(pd.DataFrame(rincian_data))

        # 3. Grafik Titik Koordinat & Vektor 2D
        st.subheader("f. Visualisasi Vektor & Titik Koordinat (2D Space)")

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

            # Gambar Garis Vektor dari Titik Pusat (0,0)
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
