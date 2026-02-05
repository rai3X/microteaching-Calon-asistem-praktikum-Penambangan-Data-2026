import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'form_data' not in st.session_state:
    st.session_state.form_data = None

st.set_page_config(
    page_title="Prediksi Kepuasan Pengguna E-Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background - Dark */
    .stApp {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    
    /* Sidebar dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: #e0e0e0 !important;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .result-card-puas {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    
    .result-card-tidak-puas {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);
    }
    
    .model-info-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
    
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .insight-card.good {
        border-left-color: #10b981;
    }
    
    .insight-card.warning {
        border-left-color: #f59e0b;
    }
    
    .insight-card.bad {
        border-left-color: #ef4444;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button[type="primary"],
    .stButton > button:first-of-type {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .stButton > button[type="primary"]:hover,
    .stButton > button:first-of-type:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stButton > button:not([type="primary"]) {
        background: rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:not([type="primary"]):hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Section header */
    .section-header {
        color: #e0e0e0;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metric display */
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #e0e0e0; font-size: 2.5rem; margin-bottom: 0.5rem;'>
            📊 Prediksi Tingkat Kepuasan Pengguna
        </h1>
        <p style='color: #9ca3af; font-size: 1.1rem;'>
            Aplikasi Pembelajaran Daring - Sistem Prediksi Berbasis Machine Learning
        </p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem;'>
            <h2 style='color: #e0e0e0;'>⚙️ Konfigurasi</h2>
        </div>
    """, unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "🤖 Pilih Algoritma Model",
        ["Logistic Regression", "Naive Bayes", "Decision Tree", "K-Nearest Neighbors"],
        help="Pilih satu algoritma machine learning untuk prediksi"
    )
    
    model_descriptions = {
        "Logistic Regression": {
            "icon": "📊",
            "description": "Model linear untuk klasifikasi biner menggunakan fungsi logistik. Cocok untuk data dengan hubungan linear antara fitur dan target.",
            "strength": "Interpretable, cepat, stabil",
            "use_case": "Baseline model untuk klasifikasi biner"
        },
        "Naive Bayes": {
            "icon": "🎲",
            "description": "Model probabilistik berdasarkan teorema Bayes dengan asumsi independensi fitur. Efektif untuk data dengan distribusi normal.",
            "strength": "Cepat, baik untuk data kecil, output probabilistik",
            "use_case": "Model probabilistik untuk klasifikasi"
        },
        "Decision Tree": {
            "icon": "🌳",
            "description": "Model non-linear yang membangun pohon keputusan berdasarkan aturan if-else. Dapat menangani hubungan non-linear antar fitur.",
            "strength": "Mudah diinterpretasi, menangani non-linearitas",
            "use_case": "Model non-linear untuk klasifikasi"
        },
        "K-Nearest Neighbors": {
            "icon": "📍",
            "description": "Model berbasis jarak yang memprediksi berdasarkan k tetangga terdekat. Tidak memerlukan asumsi distribusi data.",
            "strength": "Sederhana, tidak perlu asumsi, baik untuk non-linear",
            "use_case": "Model berbasis instance untuk klasifikasi"
        }
    }
    
    model_info = model_descriptions[selected_model]
    st.markdown("---")
    st.markdown(f"""
        <div class="model-info-card">
            <h3 style='color: #10b981; margin-bottom: 1rem;'>
                {model_info['icon']} {selected_model}
            </h3>
            <p style='color: #d1d5db; line-height: 1.6; margin-bottom: 1rem;'>
                {model_info['description']}
            </p>
            <p style='color: #9ca3af; font-size: 0.9rem;'>
                <strong>Kelebihan:</strong> {model_info['strength']}
            </p>
        </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        models = {
            'Logistic Regression': joblib.load('logistic_regression.joblib'),
            'Naive Bayes': joblib.load('naive_bayes.joblib'),
            'Decision Tree': joblib.load('decision_tree.joblib'),
            'K-Nearest Neighbors': joblib.load('knn.joblib')
        }
        scaler = joblib.load('scaler.joblib')
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"❌ **Error:** File model tidak ditemukan: {str(e)}")
        st.info("💡 **Tips:** Jalankan notebook `Technical_Test_Datmin_Terstruktur.ipynb` untuk melatih model terlebih dahulu.")
        return None, None
    except Exception as e:
        st.error(f"❌ **Error:** {str(e)}")
        st.info("💡 **Tips:** Pastikan semua file model (.joblib) berada di folder yang sama dengan app.py")
        return None, None

models, scaler = load_models()

if models and scaler:
    st.markdown('<div class="section-header">📝 Input Data Pengguna</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
            <div class="prediction-card">
                <h3 style='color: #e0e0e0; margin-bottom: 1rem;'>👤 Profil Pengguna</h3>
            </div>
        """, unsafe_allow_html=True)
        
        jenis_kelamin = st.radio(
            "Jenis Kelamin",
            ["Laki-laki", "Perempuan"],
            horizontal=True
        )
        
        usia = st.number_input(
            "Usia",
            min_value=18,
            max_value=100,
            value=25,
            step=1
        )
        
        frekuensi_login = st.slider(
            "Frekuensi Login (per minggu)",
            min_value=1,
            max_value=7,
            value=3,
            step=1,
            help="1 = Sekali per minggu | 7 = Setiap hari"
        )
        st.caption(f"📊 Nilai: {frekuensi_login} kali/minggu")
        
        durasi_penggunaan = st.slider(
            "Durasi Penggunaan (jam/hari)",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.1,
            format="%.1f",
            help="0 = Tidak digunakan | 24 = Sepanjang hari"
        )
        st.caption(f"⏱️ Nilai: {durasi_penggunaan:.1f} jam/hari")
    
    with col2:
        st.markdown("""
            <div class="prediction-card">
                <h3 style='color: #e0e0e0; margin-bottom: 1rem;'>⭐ Penilaian Fitur</h3>
        """, unsafe_allow_html=True)
        
        kualitas_materi = st.slider(
            "Kualitas Materi Pembelajaran",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="1=Sangat Buruk | 3=Cukup | 5=Sangat Baik"
        )
        st.caption(f"📚 Skor: {kualitas_materi}/5 {'⭐' * kualitas_materi}")
        
        kemudahan_penggunaan = st.slider(
            "Kemudahan Penggunaan",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="1=Sangat Sulit | 3=Cukup Mudah | 5=Sangat Mudah"
        )
        st.caption(f"🎯 Skor: {kemudahan_penggunaan}/5 {'⭐' * kemudahan_penggunaan}")
        
        stabilitas_aplikasi = st.slider(
            "Stabilitas Aplikasi",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="1=Sangat Tidak Stabil | 3=Cukup Stabil | 5=Sangat Stabil"
        )
        st.caption(f"🛡️ Skor: {stabilitas_aplikasi}/5 {'⭐' * stabilitas_aplikasi}")
        
        interaksi_pengajar = st.slider(
            "Interaksi dengan Pengajar",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="1=Sangat Buruk | 3=Cukup Baik | 5=Sangat Baik"
        )
        st.caption(f"👨‍🏫 Skor: {interaksi_pengajar}/5 {'⭐' * interaksi_pengajar}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔍 PREDIKSI KEPUASAN", type="primary", use_container_width=True)
    
    if predict_button:
        if usia < 18 or usia > 100:
            st.error("❌ Usia harus berada di antara 18-100 tahun!")
            st.stop()
        if durasi_penggunaan < 0:
            st.error("❌ Durasi penggunaan tidak boleh negatif!")
            st.stop()
        
        st.session_state.form_submitted = True
        st.session_state.form_data = {
            'jenis_kelamin': jenis_kelamin,
            'usia': usia,
            'frekuensi_login': frekuensi_login,
            'durasi_penggunaan': durasi_penggunaan,
            'kualitas_materi': kualitas_materi,
            'kemudahan_penggunaan': kemudahan_penggunaan,
            'stabilitas_aplikasi': stabilitas_aplikasi,
            'interaksi_pengajar': interaksi_pengajar
        }
        
        with st.spinner("🔄 Memproses prediksi..."):
            gender_mapping = {"Laki-laki": 0, "Perempuan": 1}
            jenis_kelamin_encoded = gender_mapping[jenis_kelamin]
            
            input_data = pd.DataFrame({
                'usia': [usia],
                'durasi_penggunaan': [durasi_penggunaan],
                'frekuensi_login': [frekuensi_login],
                'kualitas_materi': [kualitas_materi],
                'kemudahan_penggunaan': [kemudahan_penggunaan],
                'stabilitas_aplikasi': [stabilitas_aplikasi],
                'interaksi_pengajar': [interaksi_pengajar],
                'jenis_kelamin_encoded': [jenis_kelamin_encoded]
            })
            
            feature_order = ['usia', 'durasi_penggunaan', 'frekuensi_login', 'kualitas_materi',
                           'kemudahan_penggunaan', 'stabilitas_aplikasi', 'interaksi_pengajar', 'jenis_kelamin_encoded']
            input_data = input_data[feature_order]
            
            model = models[selected_model]
            
            if selected_model == 'Decision Tree':
                pred = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
            else:
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
            
            proba_puas = proba[1]
            proba_tidak_puas = proba[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">� Ringkasan Input</div>', unsafe_allow_html=True)
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Jenis Kelamin</div>
                    <div style="font-size: 1.2rem; color: #10b981; font-weight: 600;">{st.session_state.form_data['jenis_kelamin']}</div>
                </div>
            """, unsafe_allow_html=True)
        with summary_col2:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Usia</div>
                    <div style="font-size: 1.2rem; color: #10b981; font-weight: 600;">{st.session_state.form_data['usia']} tahun</div>
                </div>
            """, unsafe_allow_html=True)
        with summary_col3:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Durasi/Hari</div>
                    <div style="font-size: 1.2rem; color: #10b981; font-weight: 600;">{st.session_state.form_data['durasi_penggunaan']:.1f}h</div>
                </div>
            """, unsafe_allow_html=True)
        with summary_col4:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Login/Minggu</div>
                    <div style="font-size: 1.2rem; color: #10b981; font-weight: 600;">{st.session_state.form_data['frekuensi_login']}x</div>
                </div>
            """, unsafe_allow_html=True)
        
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)
        with score_col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Kualitas Materi</div>
                    <div style="font-size: 1.2rem; color: #f59e0b; font-weight: 600;">{st.session_state.form_data['kualitas_materi']}/5 ⭐</div>
                </div>
            """, unsafe_allow_html=True)
        with score_col2:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Kemudahan</div>
                    <div style="font-size: 1.2rem; color: #f59e0b; font-weight: 600;">{st.session_state.form_data['kemudahan_penggunaan']}/5 ⭐</div>
                </div>
            """, unsafe_allow_html=True)
        with score_col3:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Stabilitas</div>
                    <div style="font-size: 1.2rem; color: #f59e0b; font-weight: 600;">{st.session_state.form_data['stabilitas_aplikasi']}/5 ⭐</div>
                </div>
            """, unsafe_allow_html=True)
        with score_col4:
            st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.3rem;">Interaksi Pengajar</div>
                    <div style="font-size: 1.2rem; color: #f59e0b; font-weight: 600;">{st.session_state.form_data['interaksi_pengajar']}/5 ⭐</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">�📊 Hasil Prediksi</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            if pred == 1:
                st.markdown(f"""
                    <div class="result-card-puas">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">😊</div>
                        <h2 style="font-size: 2.5rem; margin: 1rem 0; color: white;">PUAS</h2>
                        <p style="font-size: 1.2rem; opacity: 0.95; margin: 0.5rem 0;">
                            Pengguna diprediksi merasa <strong>PUAS</strong> dengan aplikasi
                        </p>
                        <div style="margin-top: 1.5rem; font-size: 1rem; opacity: 0.9;">
                            Probabilitas: <strong>{proba_puas*100:.2f}%</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-tidak-puas">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">😞</div>
                        <h2 style="font-size: 2.5rem; margin: 1rem 0; color: white;">TIDAK PUAS</h2>
                        <p style="font-size: 1.2rem; opacity: 0.95; margin: 0.5rem 0;">
                            Pengguna diprediksi merasa <strong>TIDAK PUAS</strong> dengan aplikasi
                        </p>
                        <div style="margin-top: 1.5rem; font-size: 1rem; opacity: 0.9;">
                            Probabilitas: <strong>{proba_tidak_puas*100:.2f}%</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="prediction-card">
                    <h3 style='color: #e0e0e0; margin-bottom: 1rem;'>📈 Probabilitas</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #10b981;">😊 Puas</span>
                        <span style="color: #e0e0e0; font-weight: 600;">{proba_puas*100:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); height: 100%; width: {proba_puas*100}%;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #f59e0b;">😞 Tidak Puas</span>
                        <span style="color: #e0e0e0; font-weight: 600;">{proba_tidak_puas*100:.1f}%</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); height: 100%; width: {proba_tidak_puas*100}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="prediction-card" style="margin-top: 1rem;">
                    <h4 style='color: #e0e0e0; margin-bottom: 0.5rem;'>🤖 Model</h4>
                    <p style='color: #10b981; font-weight: 600;'>{selected_model}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Grafik Probabilitas</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        categories = ['Tidak Puas', 'Puas']
        probabilities = [proba_tidak_puas * 100, proba_puas * 100]
        colors = ['#f59e0b', '#10b981']
        
        bars = ax.bar(categories, probabilities, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        ax.set_ylabel('Probabilitas (%)', fontsize=12, color='#e0e0e0', fontweight='bold')
        ax.set_title(f'Probabilitas Prediksi - {selected_model}', fontsize=14, fontweight='bold', color='#e0e0e0', pad=20)
        ax.set_ylim(0, 100)
        ax.tick_params(colors='#e0e0e0')
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['top'].set_color('#1a1a1a')
        ax.spines['right'].set_color('#1a1a1a')
        ax.spines['left'].set_color('#e0e0e0')
        ax.grid(True, alpha=0.2, axis='y', color='#e0e0e0')
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prob:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12, color='#e0e0e0')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Insight & Rekomendasi</div>', unsafe_allow_html=True)
        
        insights = []
        
        if kualitas_materi >= 4:
            insights.append({
                'title': 'Kualitas Materi',
                'score': f'{kualitas_materi}/5',
                'status': 'good',
                'message': 'Konten pembelajaran sangat baik',
                'recommendation': 'Pertahankan dan tingkatkan dengan materi terbaru'
            })
        elif kualitas_materi >= 3:
            insights.append({
                'title': 'Kualitas Materi',
                'score': f'{kualitas_materi}/5',
                'status': 'warning',
                'message': 'Konten pembelajaran cukup baik',
                'recommendation': 'Tingkatkan variasi dan kualitas konten'
            })
        else:
            insights.append({
                'title': 'Kualitas Materi',
                'score': f'{kualitas_materi}/5',
                'status': 'bad',
                'message': 'Perlu peningkatan kualitas konten',
                'recommendation': 'Review dan update materi pembelajaran secara berkala'
            })
        
        if stabilitas_aplikasi >= 4:
            insights.append({
                'title': 'Stabilitas Aplikasi',
                'score': f'{stabilitas_aplikasi}/5',
                'status': 'good',
                'message': 'Aplikasi stabil dan handal',
                'recommendation': 'Pertahankan performa dan monitoring berkala'
            })
        elif stabilitas_aplikasi >= 3:
            insights.append({
                'title': 'Stabilitas Aplikasi',
                'score': f'{stabilitas_aplikasi}/5',
                'status': 'warning',
                'message': 'Perlu optimasi performa',
                'recommendation': 'Cache optimization dan CDN untuk performa lebih baik'
            })
        else:
            insights.append({
                'title': 'Stabilitas Aplikasi',
                'score': f'{stabilitas_aplikasi}/5',
                'status': 'bad',
                'message': 'Perlu perbaikan signifikan',
                'recommendation': 'Perbaiki bug dan optimasi kode untuk stabilitas'
            })
        
        if interaksi_pengajar >= 4:
            insights.append({
                'title': 'Interaksi Pengajar',
                'score': f'{interaksi_pengajar}/5',
                'status': 'good',
                'message': 'Interaksi dengan pengajar sangat baik',
                'recommendation': 'Pertahankan komunikasi yang efektif'
            })
        elif interaksi_pengajar >= 3:
            insights.append({
                'title': 'Interaksi Pengajar',
                'score': f'{interaksi_pengajar}/5',
                'status': 'warning',
                'message': 'Perlu lebih responsif',
                'recommendation': 'Gunakan chatbot untuk respons cepat dan forum diskusi'
            })
        else:
            insights.append({
                'title': 'Interaksi Pengajar',
                'score': f'{interaksi_pengajar}/5',
                'status': 'bad',
                'message': 'Interaksi perlu ditingkatkan',
                'recommendation': 'Tingkatkan frekuensi komunikasi dan feedback'
            })
        
        if kemudahan_penggunaan >= 4:
            insights.append({
                'title': 'Kemudahan Penggunaan',
                'score': f'{kemudahan_penggunaan}/5',
                'status': 'good',
                'message': 'Aplikasi mudah digunakan',
                'recommendation': 'Pertahankan UX yang intuitif'
            })
        elif kemudahan_penggunaan >= 3:
            insights.append({
                'title': 'Kemudahan Penggunaan',
                'score': f'{kemudahan_penggunaan}/5',
                'status': 'warning',
                'message': 'Perlu perbaikan UI/UX',
                'recommendation': 'Sederhanakan navigasi dan tambahkan tutorial'
            })
        else:
            insights.append({
                'title': 'Kemudahan Penggunaan',
                'score': f'{kemudahan_penggunaan}/5',
                'status': 'bad',
                'message': 'Perlu perbaikan signifikan',
                'recommendation': 'Redesign UI/UX yang lebih intuitif dan user-friendly'
            })
        
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            with cols[idx % 2]:
                icon = "✅" if insight['status'] == 'good' else "⚠️" if insight['status'] == 'warning' else "❌"
                st.markdown(f"""
                    <div class="insight-card {insight['status']}">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem;">{icon}</span>
                            <strong style="color: #e0e0e0;">{insight['title']}</strong>
                        </div>
                        <div style="color: #d1d5db; margin-bottom: 0.5rem;">
                            Skor: {insight['score']} - {insight['message']}
                        </div>
                        <div style="color: #9ca3af; font-size: 0.9rem;">
                            <strong>Rekomendasi:</strong> {insight['recommendation']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        overall_avg = (kualitas_materi + kemudahan_penggunaan + stabilitas_aplikasi + interaksi_pengajar) / 4
        
        conclusion_icon = "✅" if pred == 1 else "⚠️"
        conclusion_color = "#10b981" if pred == 1 else "#f59e0b"
        
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: {conclusion_color}; margin-bottom: 1rem;'>
                    {conclusion_icon} Kesimpulan Keseluruhan
                </h3>
                <p style='color: #d1d5db; line-height: 1.8;'>
                    Berdasarkan prediksi menggunakan <strong>{selected_model}</strong>, pengguna diprediksi 
                    <strong>{'PUAS' if pred == 1 else 'TIDAK PUAS'}</strong> dengan tingkat keyakinan 
                    <strong>{max(proba_puas, proba_tidak_puas)*100:.1f}%</strong>.
                </p>
                <p style='color: #9ca3af; margin-top: 1rem;'>
                    Rata-rata skor penilaian: <strong>{overall_avg:.1f}/5</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        reset_button = st.button("🔄 Reset Form untuk Prediksi Baru", use_container_width=True)
        if reset_button:
            st.session_state.form_submitted = False
            st.session_state.form_data = None
            st.rerun()
        
        st.info("💡 **Gunakan tombol 'Reset Form' untuk membuat prediksi baru dengan data yang berbeda.")

else:
    st.warning("⚠️ Model tidak ditemukan. Pastikan file model telah dibuat dengan menjalankan notebook terlebih dahulu.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.1);'>
        <p style='margin: 0.5rem 0;'>Dashboard Prediksi Kepuasan Pengguna E-Learning</p>
        <p style='margin: 0.5rem 0; font-size: 0.9rem;'>Dibangun dengan Streamlit | DASPRO Laboratory</p>
        <p style='margin: 0.5rem 0; font-size: 0.8rem; opacity: 0.7;'>v1.1 - UI/UX Improvements</p>
    </div>
""", unsafe_allow_html=True)
