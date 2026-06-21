import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { padding-top: 2rem; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem 0; }
.hero p  { font-size: 0.95rem; opacity: 0.75; margin: 0; }

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 1.5rem 0 0.75rem 0;
}

.info-box {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #0369a1;
    margin-bottom: 0.5rem;
}

.result-positive {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #f87171;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #4ade80;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; }
.result-sub   { font-size: 0.9rem; opacity: 0.75; }

.model-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.model-card h4 { font-size: 0.95rem; font-weight: 600; margin: 0 0 0.25rem 0; }
.model-card p  { font-size: 0.82rem; color: #6b7280; margin: 0; }

.prob-bar-wrap {
    background: #e5e7eb;
    border-radius: 999px;
    height: 10px;
    margin-top: 0.5rem;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 999px;
    transition: width 0.5s ease;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0f3460, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    margin-top: 1rem;
}
.stButton > button:hover {
    opacity: 0.9;
}

hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("model_diabetes.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_diabetes.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal load model: {e}")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🩺 Prediksi Diabetes</h1>
    <p>Masukkan data klinis pasien untuk mendapatkan prediksi risiko diabetes menggunakan model Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# ── Model selector ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Pilih Model</div>', unsafe_allow_html=True)
model_choice = st.selectbox(
    "Model prediksi",
    ["Logistic Regression", "Naive Bayes"],
    label_visibility="collapsed"
)

# ── Input fields ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Data Klinis Pasien</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1,
        help="Jumlah kehamilan yang pernah dialami")
    glucose = st.number_input("🩸 Glucose (mg/dL)", min_value=0, max_value=300, value=120,
        help="Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral. Normal: 70–140")
    blood_pressure = st.number_input("💉 Blood Pressure (mmHg)", min_value=0, max_value=150, value=70,
        help="Tekanan darah diastolik. Normal: 60–80 mmHg")
    skin_thickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20,
        help="Ketebalan lipatan kulit trisep. Normal: 10–40 mm")

with col2:
    insulin = st.number_input("💊 Insulin (μU/mL)", min_value=0, max_value=900, value=80,
        help="Kadar insulin serum 2 jam. Normal: 16–166 μU/mL")
    bmi = st.number_input("⚖️ BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index. Normal: 18.5–24.9")
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
        help="Fungsi riwayat diabetes keluarga. Semakin tinggi = risiko genetik lebih besar")
    age = st.number_input("🎂 Age (tahun)", min_value=1, max_value=120, value=25,
        help="Usia pasien dalam tahun")

# ── Info box ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    ℹ️ Pastikan semua nilai diisi dengan benar. Nilai 0 pada Glucose, Blood Pressure, atau BMI dapat mempengaruhi akurasi prediksi.
</div>
""", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang"):
    if not model_loaded:
        st.error("Model belum berhasil dimuat.")
    else:
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        # Probabilitas (kalau model support)
        try:
            proba = model.predict_proba(input_scaled)[0]
            prob_diabetes = proba[1]
            prob_tidak   = proba[0]
            has_proba    = True
        except:
            has_proba = False

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Hasil Prediksi</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-title">⚠️ Terindikasi Diabetes</div>
                <div class="result-sub">Model mendeteksi pola yang mengindikasikan risiko diabetes.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-title">✅ Tidak Terindikasi Diabetes</div>
                <div class="result-sub">Berdasarkan data yang dimasukkan, tidak terdeteksi risiko diabetes.</div>
            </div>
            """, unsafe_allow_html=True)

        # Probabilitas bar
        if has_proba:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Probabilitas</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🟢 Tidak Diabetes", f"{prob_tidak*100:.1f}%")
                st.progress(float(prob_tidak))
            with col_b:
                st.metric("🔴 Diabetes", f"{prob_diabetes*100:.1f}%")
                st.progress(float(prob_diabetes))

        # Ringkasan input
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Ringkasan Input</div>', unsafe_allow_html=True)
        df_summary = pd.DataFrame({
            "Parameter": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                          "Insulin", "BMI", "Diabetes Pedigree", "Age"],
            "Nilai": [pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf, age],
            "Satuan": ["kali", "mg/dL", "mmHg", "mm", "μU/mL", "kg/m²", "-", "tahun"]
        })
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("""
        <br>
        <div style="text-align:center; font-size:0.8rem; color:#9ca3af;">
            ⚠️ Hasil ini bersifat prediktif dan tidak menggantikan diagnosis medis profesional.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:0.8rem; color:#9ca3af;">
    Tubes Data Mining · Kelompok 11 · 2026
</div>
""", unsafe_allow_html=True)