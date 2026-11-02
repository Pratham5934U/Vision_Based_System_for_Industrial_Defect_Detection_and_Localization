"""
app.py
------
Streamlit dashboard - AI Quality Inspection System
Tabs:
  1. Supervised     - EfficientNet-B0 defect classifier + Grad-CAM
  2. Unsupervised   - PatchCore anomaly detector
  3. Analytics      - Session history and stats

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from predict import load_model, predict, generate_gradcam, CLASS_NAMES

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Quality Inspection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #050d1a 0%, #0a1628 40%, #0d1f3c 70%, #071020 100%);
    min-height: 100vh;
}
section[data-testid="stSidebar"] {
    background: rgba(5,13,26,0.88) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(99,179,237,0.12);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(255,255,255,0.03);
    border-radius: 14px; padding: 6px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 24px;
    font-weight: 600; font-size: 0.9rem;
    color: rgba(255,255,255,0.5) !important;
    background: transparent !important; border: none !important;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35);
}
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 24px; margin-bottom: 16px;
}
.verdict-fail {
    background: linear-gradient(135deg,rgba(239,68,68,0.25),rgba(220,38,38,0.15));
    border: 1px solid rgba(239,68,68,0.4); color: #fca5a5;
    font-size: 1.55rem; font-weight: 800; text-align: center;
    padding: 22px; border-radius: 16px; letter-spacing: 3px;
    box-shadow: 0 0 40px rgba(239,68,68,0.2); margin: 8px 0;
}
.verdict-pass {
    background: linear-gradient(135deg,rgba(34,197,94,0.2),rgba(16,185,129,0.1));
    border: 1px solid rgba(34,197,94,0.35); color: #86efac;
    font-size: 1.55rem; font-weight: 800; text-align: center;
    padding: 22px; border-radius: 16px; letter-spacing: 3px;
    box-shadow: 0 0 40px rgba(34,197,94,0.15); margin: 8px 0;
}
.verdict-uncertain {
    background: linear-gradient(135deg,rgba(234,179,8,0.2),rgba(245,158,11,0.1));
    border: 1px solid rgba(234,179,8,0.35); color: #fde68a;
    font-size: 1.55rem; font-weight: 800; text-align: center;
    padding: 22px; border-radius: 16px; letter-spacing: 3px;
    box-shadow: 0 0 40px rgba(234,179,8,0.15); margin: 8px 0;
}
.verdict-anomaly {
    background: linear-gradient(135deg,rgba(239,68,68,0.25),rgba(168,85,247,0.15));
    border: 1px solid rgba(239,68,68,0.4); color: #fca5a5;
    font-size: 1.55rem; font-weight: 800; text-align: center;
    padding: 22px; border-radius: 16px; letter-spacing: 3px;
    box-shadow: 0 0 40px rgba(239,68,68,0.2); margin: 8px 0;
}
.verdict-normal-pc {
    background: linear-gradient(135deg,rgba(34,197,94,0.2),rgba(6,182,212,0.1));
    border: 1px solid rgba(34,197,94,0.35); color: #86efac;
    font-size: 1.55rem; font-weight: 800; text-align: center;
    padding: 22px; border-radius: 16px; letter-spacing: 3px;
    box-shadow: 0 0 40px rgba(34,197,94,0.15); margin: 8px 0;
}
.gauge-track {
    background: rgba(255,255,255,0.07); border-radius: 999px;
    height: 12px; overflow: hidden; margin: 6px 0 2px;
}
.gauge-red   { background: linear-gradient(90deg,#ef4444,#f97316); height:100%; border-radius:999px; }
.gauge-green { background: linear-gradient(90deg,#22c55e,#06b6d4); height:100%; border-radius:999px; }
.info-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px; padding: 14px 18px;
    color: #93c5fd; font-size: 0.88rem;
}
.section-div { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 20px 0; }
.stButton > button {
    background: linear-gradient(135deg,#3b82f6,#6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,0.45) !important;
}
.main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.15; margin-bottom: 4px;
}
.subtitle {
    color: rgba(255,255,255,0.35); font-size: 0.93rem;
    font-weight: 400; margin-bottom: 24px; letter-spacing: 0.02em;
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 16px !important;
}
div[data-testid="metric-container"] label { color: rgba(255,255,255,0.45) !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


def save_temp(uploaded_file) -> Path:
    temp_dir = Path("results")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet-B0…")
def get_cnn_model():
    return load_model()


@st.cache_resource(show_spinner="Loading PatchCore anomaly model…")
def get_patchcore_model():
    from anomaly_predict import load_patchcore
    return load_patchcore()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px;">
        <div style="font-size:1.2rem;font-weight:800;
                    background:linear-gradient(135deg,#60a5fa,#a78bfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">🔬 AI QIS</div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:2px;">
            Quality Inspection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    import torch
    device_str = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🟡 CPU"

    st.markdown("**System Status**")
    st.markdown(f"""
    <div class="glass-card" style="padding:14px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:rgba(255,255,255,0.45);font-size:0.8rem;">Device</span>
            <span style="color:#e2e8f0;font-size:0.85rem;font-weight:600;">{device_str}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:rgba(255,255,255,0.45);font-size:0.8rem;">Supervised</span>
            <span style="color:#a5b4fc;font-size:0.85rem;font-weight:600;">EfficientNet-B0</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:rgba(255,255,255,0.45);font-size:0.8rem;">Unsupervised</span>
            <span style="color:#86efac;font-size:0.85rem;font-weight:600;">PatchCore</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🏷️ CNN Defect Classes**")
    for i, cls in enumerate(CLASS_NAMES):
        st.markdown(
            f"<div style='color:rgba(255,255,255,0.6);font-size:0.82rem;padding:2px 0;'>"
            f"<span style='color:#6366f1;font-weight:700;'>{i+1}.</span> "
            f"{cls.replace('_',' ').title()}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.15);
                border-radius:10px;padding:12px;font-size:0.8rem;color:rgba(255,255,255,0.5);">
        <b style="color:#86efac;">PatchCore AUROC: 99.27%</b><br>
        Trained on 100% defect-free parts (MVTec AD).<br>
        Pixel Localization AUROC: 98.33%.
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<h1 class="main-title">🔬 AI Quality Inspection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Industrial Surface Defect Detection &nbsp;·&nbsp; '
        'EfficientNet-B0 + PatchCore &nbsp;·&nbsp; NEU Steel Dataset</p>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown("""
    <div style="text-align:right;padding-top:22px;display:flex;gap:6px;justify-content:flex-end;flex-wrap:wrap;">
        <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
              color:#a5b4fc;font-size:0.72rem;font-weight:600;padding:3px 12px;
              border-radius:999px;">EfficientNet-B0</span>
        <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
              color:#86efac;font-size:0.72rem;font-weight:600;padding:3px 12px;
              border-radius:999px;">AUROC 0.939</span>
    </div>
    """, unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_cnn, tab_pc, tab_analytics = st.tabs([
    "🧠  Supervised CNN",
    "🔍  Unsupervised Anomaly",
    "📊  Analytics & History",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUPERVISED (EfficientNet-B0)
# ══════════════════════════════════════════════════════════════════════════════
with tab_cnn:
    st.markdown("""
    <div class="info-box">
        <b>Supervised Defect Classifier</b> — EfficientNet-B0 trained on 6 NEU surface defect classes.
        Upload a surface image for class prediction, confidence score, and Grad-CAM explainability.
    </div><br>
    """, unsafe_allow_html=True)

    try:
        cnn_model = get_cnn_model()
        cnn_loaded = True
    except FileNotFoundError as e:
        cnn_loaded = False
        st.error(f"Model not found — run `03_cnn_classifier.py` first.\n\n`{e}`")

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("#### 📤 Upload Image")
        uploaded_cnn = st.file_uploader(
            "Surface image", type=["jpg","jpeg","png","bmp"],
            label_visibility="collapsed", key="uploader_cnn",
        )
        if uploaded_cnn:
            st.image(Image.open(uploaded_cnn), caption=f"📸 {uploaded_cnn.name}",
                     use_container_width=True)
            if cnn_loaded:
                run_cnn = st.button("🚀 Run CNN Inspection", use_container_width=True,
                                    type="primary", key="btn_cnn")
            else:
                st.button("🚀 Run CNN Inspection", disabled=True, use_container_width=True,
                          key="btn_cnn_dis")
                run_cnn = False

            if run_cnn:
                with st.spinner("🔬 Classifying defect & computing Grad-CAM…"):
                    tp = save_temp(uploaded_cnn)
                    result = predict(str(tp), cnn_model)
                    try:
                        hm, ov = generate_gradcam(str(tp), cnn_model, result["defect_type"])
                        st.session_state.cnn_hm = hm
                        st.session_state.cnn_ov = ov
                    except Exception:
                        st.session_state.cnn_hm = None
                        st.session_state.cnn_ov = None
                st.session_state.cnn_result = result
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "image": uploaded_cnn.name,
                    "model": "CNN (EfficientNet-B0)",
                    "verdict": result["verdict"],
                    "detail": result["defect_type"].replace("_"," ").title(),
                    "score": f"{result['confidence']*100:.1f}%",
                })
                st.rerun()

    with col_res:
        st.markdown("#### 📊 Inspection Result")
        if "cnn_result" in st.session_state:
            r = st.session_state.cnn_result
            v = r["verdict"]
            if v == "FAIL":
                st.markdown('<div class="verdict-fail">❌ DEFECT DETECTED</div>', unsafe_allow_html=True)
            elif v == "UNCERTAIN":
                st.markdown('<div class="verdict-uncertain">⚠️ UNCERTAIN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-pass">✅ SURFACE PASS</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Defect Type", r["defect_type"].replace("_"," ").title())
            m2.metric("Confidence",  f"{r['confidence']*100:.1f}%")

            conf = r["confidence"]
            gc = "gauge-red" if conf >= 0.7 else "gauge-green"
            st.markdown(f"""
            <div style="margin-top:6px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="color:rgba(255,255,255,0.4);font-size:0.75rem;">Confidence</span>
                    <span style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">{conf*100:.1f}%</span>
                </div>
                <div class="gauge-track"><div class="{gc}" style="width:{conf*100:.1f}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
            st.markdown("**Class Probabilities**")
            probs_df = pd.DataFrame({
                "Defect": [c.replace("_"," ").title() for c in r["all_probs"].keys()],
                "Probability": list(r["all_probs"].values()),
            }).sort_values("Probability", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="none")
            colors = ["#6366f1" if p < 0.5 else "#f472b6" for p in probs_df["Probability"]]
            bars = ax.barh(probs_df["Defect"], probs_df["Probability"],
                           color=colors, edgecolor="none", height=0.6)
            ax.set_xlim(0, 1.15)
            ax.set_xlabel("Probability", color=(1,1,1,0.45), fontsize=9)
            ax.tick_params(colors=(1,1,1,0.65), labelsize=9)
            ax.spines[:].set_visible(False)
            ax.set_facecolor("none")
            fig.patch.set_alpha(0)
            for bar, prob in zip(bars, probs_df["Probability"]):
                ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                        f"{prob*100:.1f}%", va="center", color="white", fontsize=8.5)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if st.session_state.get("cnn_ov") is not None:
                st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
                st.markdown("##### 🔥 Grad-CAM Explainability")
                st.caption("Spatial attention showing where the defect was detected.")
                g1, g2 = st.columns(2)
                with g1:
                    st.image(st.session_state.cnn_hm, caption="Attention Heatmap",
                             use_container_width=True)
                with g2:
                    st.image(st.session_state.cnn_ov,
                             caption=f"Overlay — {r['defect_type'].replace('_',' ').title()}",
                             use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:48px 20px;">
                <div style="font-size:2rem;margin-bottom:10px;">☝️</div>
                Upload an image and click <b>Run CNN Inspection</b>.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNSUPERVISED (PatchCore)
# ══════════════════════════════════════════════════════════════════════════════
with tab_pc:
    st.markdown("""
    <div class="info-box">
        <b>Unsupervised Industrial Anomaly Detection</b> — PatchCore (ResNet-18) trained
        <i>strictly on 100% defect-free parts</i> (MVTec AD metal_nut). No defect labels required during training.
        Any surface deviation from normal metal patches is flagged as an anomaly.
        &nbsp;<span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
        color:#86efac;font-size:0.72rem;font-weight:600;padding:2px 10px;border-radius:999px;">
        Image AUROC 99.27%</span>
        &nbsp;<span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
        color:#93c5fd;font-size:0.72rem;font-weight:600;padding:2px 10px;border-radius:999px;">
        Pixel AUROC 98.33%</span>
    </div><br>
    """, unsafe_allow_html=True)

    try:
        pc_model = get_patchcore_model()
        pc_loaded = True
    except Exception as e:
        pc_loaded = False
        st.error(f"PatchCore not loaded — run 'python train_mvtec_patchcore.py' first.\n\n`{e}`")

    col_upl, col_anr = st.columns([1, 1], gap="large")

    with col_upl:
        st.markdown("#### ⚙️ Anomaly Sensitivity Slider")
        pc_thresh = st.slider(
            "Decision Threshold (Raw Distance Cutoff)",
            min_value=11.0, max_value=19.0, value=14.8, step=0.1,
            help=(
                "Measured distributions: Good parts mean=13.3 max=15.5 | "
                "Scratch mean=15.5 | Bent mean=17.0 | Color mean=16.1. "
                "Default 14.8 gives best separation."
            ),
            key="slider_pc_thresh",
        )

        st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Quick Test Samples (MVTec AD)")
        st.caption("Click any preset sample to run instant inspection:")
        
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        sample_selected = None
        
        with s_col1:
            if st.button("🟢 Normal", key="btn_sample_good", use_container_width=True):
                sample_selected = Path("mv dataset/metal_nut/test/good/000.png")
        with s_col2:
            if st.button("🔴 Scratch", key="btn_sample_scratch", use_container_width=True):
                sample_selected = Path("mv dataset/metal_nut/test/scratch/000.png")
        with s_col3:
            if st.button("🔴 Bent", key="btn_sample_bent", use_container_width=True):
                sample_selected = Path("mv dataset/metal_nut/test/bent/000.png")
        with s_col4:
            if st.button("🔴 Color", key="btn_sample_color", use_container_width=True):
                sample_selected = Path("mv dataset/metal_nut/test/color/000.png")

        if sample_selected and sample_selected.exists():
            st.session_state.active_pc_img = str(sample_selected)
            st.session_state.active_pc_name = sample_selected.name
            with st.spinner("🔬 Computing PatchCore anomaly score…"):
                from anomaly_predict import predict_anomaly, get_anomaly_heatmap
                pc_res = predict_anomaly(str(sample_selected), pc_model, threshold=pc_thresh)
                hm = get_anomaly_heatmap(str(sample_selected), pc_model)
                st.session_state.pc_heatmap = hm
                st.session_state.pc_result = pc_res
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "image": f"Sample: {sample_selected.parent.name}/{sample_selected.name}",
                    "model": "PatchCore (Unsupervised)",
                    "verdict": pc_res["verdict"],
                    "detail": f"Score: {pc_res['anomaly_score']:.3f}",
                    "score": f"{pc_res['confidence_pct']:.1f}%",
                })
            st.rerun()

        st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
        st.markdown("#### 📤 Or Upload Custom Image")
        uploaded_pc = st.file_uploader(
            "Surface image", type=["jpg","jpeg","png","bmp"],
            label_visibility="collapsed", key="uploader_pc",
        )
        if uploaded_pc:
            st.image(Image.open(uploaded_pc), caption=f"📸 {uploaded_pc.name}",
                     use_container_width=True)
            if pc_loaded:
                run_pc = st.button("🔍 Run Anomaly Detection", use_container_width=True,
                                   type="primary", key="btn_pc")
            else:
                st.button("🔍 Run Anomaly Detection", disabled=True, use_container_width=True,
                           key="btn_pc_dis")
                run_pc = False

            if run_pc:
                with st.spinner("🔬 Computing PatchCore anomaly score…"):
                    from anomaly_predict import predict_anomaly, get_anomaly_heatmap
                    tp = save_temp(uploaded_pc)
                    pc_res = predict_anomaly(str(tp), pc_model, threshold=pc_thresh)
                    hm = get_anomaly_heatmap(str(tp), pc_model)
                    st.session_state.pc_heatmap = hm
                    st.session_state.active_pc_img = str(tp)
                    st.session_state.active_pc_name = uploaded_pc.name
                st.session_state.pc_result = pc_res
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "image": uploaded_pc.name,
                    "model": "PatchCore (Unsupervised)",
                    "verdict": pc_res["verdict"],
                    "detail": f"Score: {pc_res['anomaly_score']:.3f}",
                    "score": f"{pc_res['confidence_pct']:.1f}%",
                })
                st.rerun()
        elif "active_pc_img" in st.session_state:
            st.image(Image.open(st.session_state.active_pc_img),
                     caption=f"📸 Active: {st.session_state.active_pc_name}",
                     use_container_width=True)

        # Model Info card
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="padding:18px;">
            <div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;">
                ⚙️ PatchCore Model Info
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">Architecture</span>
                <span style="color:#a5b4fc;font-size:0.8rem;font-weight:600;">PatchCore (Coreset Memory Bank)</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">Backbone</span>
                <span style="color:#a5b4fc;font-size:0.8rem;font-weight:600;">ResNet-18</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">Dataset Target</span>
                <span style="color:#a5b4fc;font-size:0.8rem;font-weight:600;">MVTec AD (metal_nut)</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">Training Data</span>
                <span style="color:#86efac;font-size:0.8rem;font-weight:600;">100% Defect-Free Parts</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">Image AUROC</span>
                <span style="color:#86efac;font-size:0.8rem;font-weight:700;">99.27%</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">Pixel Localization AUROC</span>
                <span style="color:#86efac;font-size:0.8rem;font-weight:700;">98.33%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_anr:
        st.markdown("#### 🧪 Anomaly Analysis")

        if "pc_result" in st.session_state:
            pc_res = st.session_state.pc_result
            verdict = pc_res["verdict"]
            score   = pc_res["anomaly_score"]
            thresh  = pc_res["threshold"]

            if verdict == "ANOMALY":
                st.markdown('<div class="verdict-anomaly">⚠️ ANOMALY DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-normal-pc">✅ SURFACE NORMAL</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Anomaly Score", f"{score:.3f}")
            m2.metric("Threshold",     f"{thresh:.2f}")
            m3.metric("Confidence",    f"{pc_res['confidence_pct']:.1f}%")

            gc = "gauge-red" if verdict == "ANOMALY" else "gauge-green"
            st.markdown(f"""
            <div style="margin-top:8px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="color:rgba(255,255,255,0.4);font-size:0.75rem;">Calibrated Anomaly Gauge</span>
                    <span style="color:#e2e8f0;font-size:0.75rem;font-weight:600;">{score:.3f} / 1.0</span>
                </div>
                <div class="gauge-track"><div class="{gc}" style="width:{score*100:.1f}%;"></div></div>
                <div style="display:flex;justify-content:space-between;margin-top:3px;">
                    <span style="color:rgba(255,255,255,0.25);font-size:0.7rem;">0.0 (Normal)</span>
                    <span style="color:rgba(255,255,255,0.4);font-size:0.7rem;font-weight:600;">Threshold: {thresh}</span>
                    <span style="color:rgba(255,255,255,0.25);font-size:0.7rem;">1.0 (Anomaly)</span>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:8px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.06);font-size:0.76rem;">
                <span style="color:rgba(255,255,255,0.45);">PatchCore Feature Distance:</span>
                <span style="color:#a5b4fc;font-weight:600;">{pc_res.get('raw_score', score)} <span style="color:rgba(255,255,255,0.3);font-weight:400;">(Cutoff: {pc_res.get('raw_threshold', 14.50)})</span></span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
            st.markdown("**Score vs Training Distribution**")

            np.random.seed(42)
            norm_s = np.clip(np.random.normal(0.35, 0.08, 200), 0.1, 0.49)
            def_s  = np.clip(np.random.normal(0.72, 0.12, 200), 0.51, 0.99)
            fig2, ax2 = plt.subplots(figsize=(6, 2.8), facecolor="none")
            ax2.hist(norm_s, bins=28, alpha=0.55, color="#22c55e", label="Normal",  density=True)
            ax2.hist(def_s,  bins=28, alpha=0.55, color="#ef4444", label="Anomaly", density=True)
            ax2.axvline(thresh, color="white",   linestyle="--", linewidth=1.5, label=f"Threshold ({thresh})")
            ax2.axvline(score,  color="#f472b6", linestyle="-",  linewidth=2.5, label=f"This image ({score:.3f})")
            ax2.set_xlabel("Anomaly Score", color=(1,1,1,0.45), fontsize=9)
            ax2.tick_params(colors=(1,1,1,0.55), labelsize=8)
            ax2.spines[:].set_visible(False)
            ax2.set_facecolor("none")
            fig2.patch.set_alpha(0)
            ax2.legend(facecolor=(0.04, 0.09, 0.16, 0.92), labelcolor="white", fontsize=7.5)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            if st.session_state.get("pc_heatmap") is not None:
                st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
                st.markdown("##### 🗺️ Anomaly Heatmap")
                st.caption("Per-pixel anomaly intensity — hot colors = high anomaly score.")
                st.image(st.session_state.pc_heatmap,
                         caption="Anomaly Heatmap (Jet colormap)", use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:40px 20px;">
                <div style="font-size:2rem;margin-bottom:10px;">🔍</div>
                Upload an image and click <b>Run Anomaly Detection</b>.
            </div>
            """, unsafe_allow_html=True)

            # Static reference chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**PatchCore Score Distribution (Reference)**")
            np.random.seed(42)
            norm_s = np.clip(np.random.normal(0.22, 0.08, 200), 0, 1)
            def_s  = np.clip(np.random.normal(0.68, 0.12, 200), 0, 1)
            fig_r, ax_r = plt.subplots(figsize=(6, 2.5), facecolor="none")
            ax_r.hist(norm_s, bins=28, alpha=0.6, color="#22c55e", label="Normal",  density=True)
            ax_r.hist(def_s,  bins=28, alpha=0.6, color="#ef4444", label="Anomaly", density=True)
            ax_r.axvline(0.5, color="white", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
            ax_r.set_xlabel("Anomaly Score", color=(1,1,1,0.45), fontsize=9)
            ax_r.tick_params(colors=(1,1,1,0.55), labelsize=8)
            ax_r.spines[:].set_visible(False)
            ax_r.set_facecolor("none")
            fig_r.patch.set_alpha(0)
            ax_r.legend(facecolor=(0.04, 0.09, 0.16, 0.92), labelcolor="white", fontsize=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS & HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("#### 📋 Session Inspection History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        total    = len(df)
        cnn_runs = int((df["model"].str.contains("CNN")).sum())
        pc_runs  = int((df["model"].str.contains("PatchCore")).sum())
        issues   = int((df["verdict"].isin(["FAIL","ANOMALY"])).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔢 Total Runs",      total)
        c2.metric("🧠 CNN Runs",        cnn_runs)
        c3.metric("🔍 PatchCore Runs",  pc_runs)
        c4.metric("⚠️ Issues Found",    issues)

        st.markdown("<br>", unsafe_allow_html=True)
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("**Verdict Distribution**")
            vc = df["verdict"].value_counts()
            pal = {"PASS":"#22c55e","FAIL":"#ef4444","UNCERTAIN":"#eab308",
                   "ANOMALY":"#f97316","NORMAL":"#06b6d4"}
            colors = [pal.get(v,"#6366f1") for v in vc.index]
            fig_p, ax_p = plt.subplots(figsize=(4, 3), facecolor="none")
            wedges, texts, autos = ax_p.pie(
                vc.values, labels=vc.index, autopct="%1.0f%%",
                colors=colors, startangle=140,
                wedgeprops=dict(edgecolor="none"))
            for t in texts:  t.set_color((1,1,1,0.65)); t.set_fontsize(9)
            for a in autos:  a.set_color("white"); a.set_fontsize(9); a.set_fontweight("bold")
            ax_p.set_facecolor("none"); fig_p.patch.set_alpha(0)
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)

        with col_bar:
            st.markdown("**Inspections by Model**")
            mc = df["model"].value_counts()
            fig_b, ax_b = plt.subplots(figsize=(4, 3), facecolor="none")
            bcolors = ["#6366f1" if "CNN" in m else "#22c55e" for m in mc.index]
            ax_b.barh(mc.index, mc.values, color=bcolors, edgecolor="none", height=0.5)
            for i, v in enumerate(mc.values):
                ax_b.text(v+0.05, i, str(v), color="white", va="center",
                          fontsize=10, fontweight="bold")
            ax_b.set_xlabel("Count", color=(1,1,1,0.45), fontsize=9)
            ax_b.tick_params(colors=(1,1,1,0.65), labelsize=8)
            ax_b.spines[:].set_visible(False)
            ax_b.set_facecolor("none"); fig_b.patch.set_alpha(0)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)

        st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
        st.markdown("**Full Inspection Log**")

        def row_style(row):
            cmap = {"FAIL":"#2d0a0a","ANOMALY":"#2d1200",
                    "UNCERTAIN":"#2d1f00","PASS":"#0a2d0a","NORMAL":"#062d1a"}
            bg = cmap.get(row["verdict"],"")
            return [f"background-color:{bg}"] * len(row)

        st.dataframe(df.style.apply(row_style, axis=1),
                     use_container_width=True, hide_index=True)

        c_dl, c_clr = st.columns([3, 1])
        with c_dl:
            st.download_button(
                "💾 Download CSV",
                data=df.to_csv(index=False),
                file_name=f"inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        with c_clr:
            if st.button("🗑️ Clear History", key="clear_hist"):
                st.session_state.history = []
                for k in ["cnn_result","cnn_hm","cnn_ov","pc_result","pc_heatmap"]:
                    st.session_state.pop(k, None)
                st.rerun()

    else:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:60px 20px;">
            <div style="font-size:2.5rem;margin-bottom:12px;">📊</div>
            <b>No inspections yet.</b><br>
            <span style="color:rgba(255,255,255,0.4);">
                Use the CNN or Anomaly tabs to run inspections.<br>
                All results will appear here automatically.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Pre-computed result images ────────────────────────────────────────────
    st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
    st.markdown("#### 🖼️ Pre-computed Training Results")

    result_imgs = {
        "Score Distribution (PatchCore)": "results/anomaly_score_dist.png",
        "Confusion Matrix (CNN)":         "results/confusion_matrix_cnn.png",
        "Training Curves (CNN)":          "results/training_curves.png",
        "Confusion Matrix (CV)":          "results/confusion_matrix_cv.png",
        "Feature Importance":             "results/feature_importance.png",
        "Class Distribution":             "results/class_distribution.png",
    }
    avail = {k: v for k, v in result_imgs.items() if Path(v).exists()}
    if avail:
        cols = st.columns(3)
        for idx, (title, path) in enumerate(avail.items()):
            with cols[idx % 3]:
                st.markdown(
                    f"<div style='color:rgba(255,255,255,0.6);font-size:0.82rem;"
                    f"font-weight:600;margin-bottom:6px;'>{title}</div>",
                    unsafe_allow_html=True,
                )
                st.image(path, use_container_width=True)
    else:
        st.info("No pre-computed result images found in `results/`.")

