"""
StyleFind AI - Interactive AI Image Clustering & Style Analysis Platform
Built with Scikit-Learn, Streamlit, and Plotly
"""
import os, time, json, joblib, numpy as np, pandas as pd, streamlit as st
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
import clustering_engine as ce

# --- Page Configuration ---
st.set_page_config(
    page_title="StyleFind AI - Image Clustering & Style Prediction",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

CLASS_ICONS = {
    "T-shirt/top": "👕",
    "Trouser": "👖",
    "Pullover": "🧥",
    "Dress": "👗",
    "Coat": "🧥",
    "Sandal": "👡",
    "Shirt": "👔",
    "Sneaker": "👟",
    "Bag": "👜",
    "Ankle boot": "👢",
}

# --- Check Prerequisites ---
if not (os.path.exists("model.joblib") and os.path.exists("assets.npz") and os.path.exists("results.json")):
    st.error("Model files not found. Please verify project directory.")
    st.stop()

# --- Cached Model & Assets Loader ---
@st.cache_resource
def load():
    model = joblib.load("model.joblib")
    A = np.load("assets.npz")
    nn = NearestNeighbors(n_neighbors=5).fit(A["cat_emb"])
    return model, {k: A[k] for k in A.files}, nn, json.load(open("results.json"))

model, A, nn, R = load()
winner = R.get("_winner", "Random Forest")
winner_results = R.get(winner, {})
winner_acc = winner_results.get("test_accuracy", 0.848)
winner_f1 = winner_results.get("test_macro_f1", 0.847)
winner_params = winner_results.get("best_params", {})

# Extract feature pipeline & Random Forest classifier
feature_pipeline = model[:-1]
rf_classifier = model.named_steps.get("clf", None)
rf_importances = rf_classifier.feature_importances_ if (rf_classifier is not None and hasattr(rf_classifier, "feature_importances_")) else None

# --- Image Preprocessing ---
def preprocess(img: Image.Image) -> np.ndarray:
    return ce.preprocess_single_image(img)

# --- Session State Management ---
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Style Prediction"
if "pred_analyzed" not in st.session_state:
    st.session_state.pred_analyzed = False
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "active_sample_idx" not in st.session_state:
    st.session_state.active_sample_idx = 0

# Clustering State
if "clustering_result" not in st.session_state:
    st.session_state.clustering_result = None
if "cluster_images_cache" not in st.session_state:
    st.session_state.cluster_images_cache = None
if "elbow_result" not in st.session_state:
    st.session_state.elbow_result = None
if "selected_k" not in st.session_state:
    st.session_state.selected_k = 3
if "selected_algo" not in st.session_state:
    st.session_state.selected_algo = "K-Means"
if "use_rf_weighting" not in st.session_state:
    st.session_state.use_rf_weighting = False
if "sample_batch_size" not in st.session_state:
    st.session_state.sample_batch_size = 25

# --- Inject Modern Dashboard CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Dark Navy Background */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #070d19 45%, #020617 100%) !important;
    color: #f1f5f9 !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px !important;
}

/* Glassmorphic Cards */
.sf-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.sf-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 16px 40px -8px rgba(99, 102, 241, 0.2);
}

.sf-hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.88) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 32px 30px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 24px;
}

/* Gradient Text */
.gradient-text {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.gradient-text-accent {
    background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Step Badges & Cards */
.step-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 14px;
    text-align: center;
    transition: all 0.22s ease;
    height: 100%;
}
.step-card:hover {
    border-color: rgba(129, 140, 248, 0.4);
    transform: translateY(-3px);
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

/* Dynamic Metric Cards */
.kpi-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 18px;
    text-align: left;
    transition: all 0.2s ease;
}
.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}
.kpi-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.kpi-val {
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 4px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #38bdf8;
    margin-top: 2px;
    font-weight: 500;
}

/* Flow Chart Component */
.flow-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    overflow-x: auto;
    padding: 14px 10px;
    gap: 8px;
    background: rgba(10, 16, 31, 0.6);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 14px;
}
.flow-step {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #e2e8f0;
    text-align: center;
    white-space: nowrap;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.flow-arrow {
    color: #818cf8;
    font-size: 0.85rem;
    font-weight: 800;
}

/* Styled Streamlit Buttons */
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(30, 41, 59, 0.6) !important;
    color: #f1f5f9 !important;
    transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div.stButton > button:hover {
    background: rgba(51, 65, 85, 0.85) !important;
    border-color: rgba(129, 140, 248, 0.5) !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.25) !important;
}
div.stButton > button[data-testid="baseButton-primary"],
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #6366f1 50%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
}
div.stButton > button[data-testid="baseButton-primary"]:hover,
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 28px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-2px) !important;
}

/* Probability Bars */
.prob-bar-bg {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 9999px;
    height: 10px;
    overflow: hidden;
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #38bdf8 0%, #6366f1 70%, #ec4899 100%);
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# TOP BAR & THREE MAIN SECTIONS NAVIGATION
# ==============================================================================
nav_col_logo, nav_col_menu = st.columns([1.1, 2.9])

with nav_col_logo:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0;">
        <span style="font-size: 1.9rem; background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 4px 8px;">✨👕</span>
        <div>
            <div style="font-size: 1.3rem; font-weight: 800; letter-spacing: -0.02em; color: #ffffff;">StyleFind <span class="gradient-text">AI</span></div>
            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">AI-Powered Image Clustering & Style Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col_menu:
    nav_btn_cols = st.columns(3)
    nav_sections = ["Style Prediction", "Image Clustering", "Algorithm Comparison"]
    icons = ["🔮", "✨", "⚖️"]
    
    for i, sec in enumerate(nav_sections):
        with nav_btn_cols[i]:
            btn_type = "primary" if st.session_state.active_nav == sec else "secondary"
            if st.button(f"{icons[i]} {sec}", key=f"top_nav_{sec}", use_container_width=True, type=btn_type):
                st.session_state.active_nav = sec
                st.rerun()

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# HERO SECTION & HORIZONTAL PIPELINE FLOWCHART
# ==============================================================================
st.markdown("""
<div class="sf-hero-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div style="max-width: 780px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(99, 102, 241, 0.18); border: 1px solid rgba(99, 102, 241, 0.35); color: #c7d2fe; padding: 5px 14px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; margin-bottom: 12px;">
                <span>✦</span> Random Forest Ensemble & Unsupervised Machine Learning
            </div>
            <h1 style="font-size: 2.35rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em; color: #ffffff; margin-bottom: 10px;">
                Discover Hidden Patterns in Images with <span class="gradient-text">AI</span>
            </h1>
            <p style="font-size: 1.0rem; color: #94a3b8; line-height: 1.6; margin-bottom: 8px;">
                StyleFind AI uses Random Forest and K-Means Machine Learning to analyze, classify, and automatically group visually similar images.
            </p>
        </div>
    </div>
    
    <!-- Attractive Horizontal Computer Vision Flow Diagram -->
    <div style="margin-top: 14px;">
        <div style="font-size: 0.72rem; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
            ⚡ End-to-End Visual Machine Learning Architecture
        </div>
        <div class="flow-container">
            <div class="flow-step">IMAGE DATA</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step">PREPROCESSING</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step">FEATURE EXTRACTION</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step">PCA</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step" style="border-color: #a855f7; color: #c084fc;">RANDOM FOREST</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step">CLUSTERING</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step">VISUALIZATION</div>
            <div class="flow-arrow">&rarr;</div>
            <div class="flow-step" style="border-color: #10b981; color: #34d399;">INSIGHTS</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Action Jump Buttons
quick_c1, quick_c2, quick_c3 = st.columns(3)
with quick_c1:
    if st.button("🔮 Predict Style", key="hero_jump_pred", use_container_width=True):
        st.session_state.active_nav = "Style Prediction"
        st.rerun()
with quick_c2:
    if st.button("✨ Cluster Images", key="hero_jump_clust", use_container_width=True):
        st.session_state.active_nav = "Image Clustering"
        st.rerun()
with quick_c3:
    if st.button("⚖️ Explore Algorithms", key="hero_jump_algo", use_container_width=True):
        st.session_state.active_nav = "Algorithm Comparison"
        st.rerun()

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: STYLE PREDICTION (Random Forest Supervised Classification)
# ==============================================================================
if st.session_state.active_nav == "Style Prediction":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">
            <span>🔮</span> Supervised Random Forest Classifier (100 Trees)
        </div>
        <h2 style="font-size: 2.0rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 4px;">Style Prediction</h2>
        <p style="font-size: 0.92rem; color: #94a3b8;">
            Upload single image &rarr; Preprocessing &rarr; Feature Extraction &rarr; Random Forest Pipeline &rarr; Style Prediction &rarr; Confidence Score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    input_mode_tabs = st.radio(
        "Choose Input Source",
        ["📤 Upload My Own Image", "📷 Use Camera", "🧪 Unseen Test Sample"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_to_process = None
    true_label_name = None

    if input_mode_tabs.startswith("📤"):
        st.markdown("""
        <div class="sf-card" style="text-align: center; border: 2px dashed rgba(99, 102, 241, 0.35); background: rgba(15, 23, 42, 0.4); padding: 24px 20px; margin-top: 10px;">
            <div style="font-size: 2.2rem; margin-bottom: 6px;">📸</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">Drop your clothing image here</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Supports JPG, JPEG, PNG, WEBP &bull; Evaluated via Random Forest Model (100 Trees)</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="pred_uploader"
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)

    elif input_mode_tabs.startswith("📷"):
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-top: 10px;'>Capture a photo of your apparel item using your webcam:</p>", unsafe_allow_html=True)
        camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera_file is not None:
            image_to_process = Image.open(camera_file)

    else:
        st.markdown("""
        <div class="sf-card" style="margin-top: 10px; padding: 18px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">Unseen Fashion-MNIST Test Split</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 8px;">Pick any sample from the 300 test images (never seen during model training):</div>
        """, unsafe_allow_html=True)

        sample_idx = st.slider(
            "Sample Selector",
            min_value=0,
            max_value=len(A["test_img"]) - 1,
            value=st.session_state.active_sample_idx,
            label_visibility="collapsed",
            key="pred_slider"
        )
        st.session_state.active_sample_idx = sample_idx
        raw_test_arr = A["test_img"][sample_idx].reshape(28, 28)
        image_to_process = Image.fromarray(raw_test_arr.astype("uint8"), mode="L")
        true_label_name = CLASSES[A["test_lab"][sample_idx]]

        st.markdown(f"""
        <div style="display: flex; gap: 8px; align-items: center; margin-top: 6px;">
            <span style="font-size: 0.78rem; color: #94a3b8;">Ground Truth Label:</span>
            <span style="font-size: 0.78rem; font-weight: 700; color: #10b981; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); padding: 2px 8px; border-radius: 6px;">
                {CLASS_ICONS.get(true_label_name, '👕')} {true_label_name} (Sample #{sample_idx})
            </span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    if image_to_process is not None:
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        preview_col, action_col = st.columns([1.1, 1.9], gap="medium")

        with preview_col:
            st.markdown("""
            <div class="sf-card" style="text-align: center; padding: 16px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">Input Image</div>
            """, unsafe_allow_html=True)
            st.image(image_to_process, width=220)
            st.markdown("</div>", unsafe_allow_html=True)

        with action_col:
            st.markdown(f"""
            <div class="sf-card" style="padding: 22px;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">Ready for Inference</div>
                <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5; margin-bottom: 16px;">
                    Runs image standardization, 50-D PCA feature projection, and queries the tuned <strong>Random Forest Ensemble</strong> ({winner_acc:.1%} accuracy across 100 trees).
                </p>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns([1.3, 1.0])
            with btn_col1:
                analyze_clicked = st.button("✨ Predict Style", use_container_width=True, type="primary", key="btn_run_pred")
            with btn_col2:
                reset_clicked = st.button("🔄 Reset Image", use_container_width=True, key="btn_reset_pred")

            st.markdown("</div>", unsafe_allow_html=True)

            if reset_clicked:
                st.session_state.pred_analyzed = False
                st.session_state.prediction_data = None
                st.rerun()

            if analyze_clicked or st.session_state.pred_analyzed:
                if analyze_clicked:
                    with st.spinner("Random Forest is evaluating tree voting consensus..."):
                        time.sleep(0.3)
                        arr_28 = preprocess(image_to_process)
                        x = arr_28.reshape(1, -1)
                        proba = model.predict_proba(x)[0]
                        top_indices = np.argsort(proba)[::-1][:3]
                        idx_neighbors = nn.kneighbors(model[:-1].transform(x), return_distance=False)[0]

                        st.session_state.prediction_data = {
                            "arr_28": arr_28,
                            "proba": proba,
                            "top_indices": top_indices,
                            "idx_neighbors": idx_neighbors,
                            "true_label": true_label_name,
                        }
                        st.session_state.pred_analyzed = True

        # --- Prediction Result Card ---
        if st.session_state.pred_analyzed and st.session_state.prediction_data is not None:
            pdata = st.session_state.prediction_data
            top_class_idx = pdata["top_indices"][0]
            top_class_name = CLASSES[top_class_idx]
            top_confidence = pdata["proba"][top_class_idx]
            top_icon = CLASS_ICONS.get(top_class_name, "👕")

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            res_col_left, res_col_right = st.columns([1.1, 1.9], gap="large")

            with res_col_left:
                st.markdown(f"""
                <div class="sf-card" style="text-align: center; border: 1px solid rgba(16, 185, 129, 0.4); background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.85) 100%);">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">Predicted Style</div>
                    <div style="font-size: 3.2rem; margin-bottom: 4px;">{top_icon}</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">{top_class_name}</div>
                    <div style="display: inline-block; margin-top: 8px; font-size: 0.95rem; font-weight: 700; color: #34d399; background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 14px; border-radius: 9999px;">
                        {top_confidence:.1%} Random Forest Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="sf-card" style="text-align: center; padding: 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">What the Model Sees</div>
                """, unsafe_allow_html=True)
                st.image(pdata["arr_28"].astype("uint8"), width=150, caption="28x28 normalized grayscale")
                st.markdown("</div>", unsafe_allow_html=True)

            with res_col_right:
                st.markdown("""
                <div class="sf-card">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 14px;">
                        Random Forest Probability Distribution Across Classes
                    </div>
                """, unsafe_allow_html=True)

                for rank, c_idx in enumerate(pdata["top_indices"]):
                    c_name = CLASSES[c_idx]
                    c_prob = pdata["proba"][c_idx]
                    c_icon = CLASS_ICONS.get(c_name, "👕")
                    pct_width = int(c_prob * 100)
                    st.markdown(f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px;">
                            <span>{c_icon} {c_name}</span>
                            <span style="font-family: monospace; color: #38bdf8;">{c_prob:.1%}</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: {pct_width}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # Nearest Neighbours from Catalogue
            st.markdown("""
            <div class="sf-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">Similar Items from the Catalogue</div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">5 nearest neighbors indexed in 50-D PCA feature space</div>
                    </div>
                    <span style="font-size: 0.72rem; color: #38bdf8; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 10px; border-radius: 9999px; font-weight: 600;">
                        k-NN (k=5)
                    </span>
                </div>
            """, unsafe_allow_html=True)

            nn_cols = st.columns(5)
            for col, cat_idx in zip(nn_cols, pdata["idx_neighbors"]):
                sim_label = CLASSES[A["cat_lab"][cat_idx]]
                sim_icon = CLASS_ICONS.get(sim_label, "👕")
                with col:
                    st.image(A["cat_img"][cat_idx].reshape(28, 28), width=95)
                    st.markdown(f"<div style='font-size: 0.75rem; font-weight: 600; text-align: center; color: #cbd5e1; margin-top: 2px;'>{sim_icon} {sim_label}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # --- RANDOM FOREST ARCHITECTURE & FEATURES SECTION ---
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: #34d399; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">
            <span>🌲</span> Model Architecture & Explainability
        </div>
        <h3 style="font-size: 1.5rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 2px;">
            Features in the Random Forest Technique
        </h3>
        <p style="font-size: 0.85rem; color: #94a3b8;">
            Deep-dive into the architectural features, bagging mechanisms, and feature importances powering StyleFind AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    rf_col_left, rf_col_right = st.columns([1.2, 1.8], gap="large")

    with rf_col_left:
        st.markdown("""
        <div class="sf-card" style="border: 1px solid rgba(16, 185, 129, 0.3); height: 100%;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #10b981; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                Ensemble Specifications
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-bottom: 12px;">
                Tuned Random Forest Model
            </div>
            
            <div style="margin-bottom: 12px; padding: 10px 14px; background: rgba(30, 41, 59, 0.5); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">ENSEMBLE SIZE</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff;">100 Decision Trees</div>
                <div style="font-size: 0.7rem; color: #38bdf8;">Trained on bootstrap data subsets</div>
            </div>

            <div style="margin-bottom: 12px; padding: 10px 14px; background: rgba(30, 41, 59, 0.5); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">MAXIMUM TREE DEPTH</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff;">20 Split Levels</div>
                <div style="font-size: 0.7rem; color: #10b981;">Prevents tree memorization & overfitting</div>
            </div>

            <div style="margin-bottom: 12px; padding: 10px 14px; background: rgba(30, 41, 59, 0.5); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">FEATURE SUBSPACE SAMPLING</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff;">Square Root of 50 Features (~7)</div>
                <div style="font-size: 0.7rem; color: #c084fc;">Decorrelates individual trees for stability</div>
            </div>

            <div style="padding: 10px 14px; background: rgba(30, 41, 59, 0.5); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">SPLITTING CRITERION</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff;">Gini Impurity</div>
                <div style="font-size: 0.7rem; color: #f59e0b;">Maximizes class purity at each decision node</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rf_col_right:
        st.markdown("""
        <div class="sf-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                Top PCA Feature Importances (MDI - Mean Decrease in Impurity)
            </div>
            <p style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 12px;">
                Relative contribution of top latent PCA features across all 100 decision trees in the ensemble:
            </p>
        """, unsafe_allow_html=True)

        if rf_classifier is not None and hasattr(rf_classifier, "feature_importances_"):
            importances = rf_classifier.feature_importances_
            top_feat_indices = importances.argsort()[::-1][:8]
            df_rf_imp = pd.DataFrame({
                "PCA Component": [f"Component {i+1}" for i in top_feat_indices],
                "Importance Weight": importances[top_feat_indices]
            })

            fig_rf_imp = px.bar(
                df_rf_imp,
                x="Importance Weight",
                y="PCA Component",
                orientation="h",
                color="Importance Weight",
                color_continuous_scale="Purples",
                text_auto=".3f"
            )
            fig_rf_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                margin=dict(l=20, r=20, t=10, b=20),
                height=260,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_rf_imp, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 6 Core Technique Cards
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    rf_feat_cols1 = st.columns(3)
    rf_tech_features = [
        ("1. Bootstrap Aggregation (Bagging)", "Each of the 100 trees is trained on an independently sampled bootstrap subset of the 60,000 training images, drastically reducing variance without increasing bias."),
        ("2. Random Feature Subspace", "At every branch node, only a random subset of ~7 orthogonal PCA features is considered for splitting, ensuring individual trees remain decorrelated and diverse."),
        ("3. Gini Impurity Optimization", "Evaluates node split thresholds by measuring the probability of incorrect labeling, guaranteeing that child leaves contain high-purity clothing categories."),
        ("4. Soft Voting Consensus", "Final classification confidence is calculated by averaging the continuous posterior probability distributions across all 100 estimators for reliable confidence scoring."),
        ("5. Non-Linear Decision Boundaries", "Unlike linear classifiers, Random Forest captures intricate non-linear relationships between complex fashion silhouettes (such as subtle lapels or sleeves)."),
        ("6. Robustness to Noise & Outliers", "Individual noisy pixel artifacts or atypical photo angles are neutralized through ensemble averaging, providing high stability on real user uploads.")
    ]

    for i in range(3):
        with rf_feat_cols1[i]:
            title, desc = rf_tech_features[i]
            st.markdown(f"""
            <div class="sf-card" style="height: 100%;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    rf_feat_cols2 = st.columns(3)
    for i in range(3, 6):
        with rf_feat_cols2[i - 3]:
            title, desc = rf_tech_features[i]
            st.markdown(f"""
            <div class="sf-card" style="height: 100%;">
                <div style="font-size: 0.92rem; font-weight: 700; color: #a855f7; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: AI IMAGE CLUSTERING (Enhanced with Random Forest Intelligence)
# ==============================================================================
elif st.session_state.active_nav == "Image Clustering":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.35); color: #c084fc; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">
            <span>✨</span> Unsupervised Clustering & Random Forest Style Labeling
        </div>
        <h2 style="font-size: 2.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 4px;">AI Image Clustering</h2>
        <p style="font-size: 0.95rem; color: #94a3b8;">
            Upload multiple images and let AI automatically group visually similar images with Random Forest validation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- 7-STEP NUMBERED PROCESS WORKFLOW ---
    step_cols = st.columns(7)
    steps_data = [
        ("1", "Upload Images", "Select 10-50 photos or quick batch"),
        ("2", "Algorithm", "Choose K-Means / Hierarchical"),
        ("3", "Choose K", "Set number of clusters (2-8)"),
        ("4", "Extract Features", "Standardize & PCA reduction"),
        ("5", "Run Clustering", "Fit cluster model on vectors"),
        ("6", "Visualize 2D", "Interactive PCA scatter plot"),
        ("7", "RF Analysis", "Validate cluster purity with RF"),
    ]
    for col, (num, title, desc) in zip(step_cols, steps_data):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #ffffff; margin-bottom: 2px;">{title}</div>
                <div style="font-size: 0.68rem; color: #94a3b8; line-height: 1.3;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # --- REAL-TIME DYNAMIC DASHBOARD STRIP ---
    kpi_res = st.session_state.clustering_result
    n_uploaded = len(st.session_state.cluster_images_cache) if st.session_state.cluster_images_cache is not None else 0
    cur_algo = kpi_res["algorithm"] if kpi_res else st.session_state.selected_algo
    cur_k = kpi_res["n_clusters"] if kpi_res else st.session_state.selected_k
    cur_sil = f"{kpi_res['silhouette_score']:.3f}" if kpi_res else "—"
    cur_time = f"{kpi_res['execution_time_ms']:.1f} ms" if kpi_res else "—"

    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Images Uploaded</div>
            <div class="kpi-val">{n_uploaded}</div>
            <div class="kpi-sub">{"Ready in Memory" if n_uploaded > 0 else "Select Dataset"}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Algorithm</div>
            <div class="kpi-val" style="font-size: 1.25rem;">{cur_algo}</div>
            <div class="kpi-sub" style="color: #a855f7;">{"Fitted" if kpi_res else "Selected"}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Clusters (K)</div>
            <div class="kpi-val" style="color: #38bdf8;">{cur_k}</div>
            <div class="kpi-sub">Groups Formed</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Silhouette Score</div>
            <div class="kpi-val" style="color: #10b981;">{cur_sil}</div>
            <div class="kpi-sub" style="color: #10b981;">Cluster Cohesion</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Processing Time</div>
            <div class="kpi-val" style="color: #f59e0b;">{cur_time}</div>
            <div class="kpi-sub" style="color: #f59e0b;">Scikit-Learn Latency</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- CLUSTERING CONTROLS & DATASET SELECTOR ---
    ctrl_col_left, ctrl_col_right = st.columns([1.1, 1.9], gap="medium")

    with ctrl_col_left:
        st.markdown("""
        <div class="sf-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 14px;">
                ⚙️ Configuration & Algorithm
            </div>
        """, unsafe_allow_html=True)

        algo_options = ["K-Means", "Hierarchical Clustering", "DBSCAN"]
        chosen_algo = st.selectbox("Select Algorithm", algo_options, index=0)
        st.session_state.selected_algo = chosen_algo

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 6px;'>Number of Clusters (K):</div>", unsafe_allow_html=True)

        # Cluster preset quick buttons
        q_cols = st.columns(4)
        for i, k_val in enumerate([2, 3, 4, 5]):
            with q_cols[i]:
                is_curr = (st.session_state.selected_k == k_val)
                if st.button(f"[{k_val}]", key=f"btn_k_{k_val}", use_container_width=True, type="primary" if is_curr else "secondary"):
                    st.session_state.selected_k = k_val
                    st.rerun()

        chosen_k = st.slider("Fine-tune K (2-8)", min_value=2, max_value=8, value=st.session_state.selected_k, label_visibility="collapsed")
        if chosen_k != st.session_state.selected_k:
            st.session_state.selected_k = chosen_k

        # Random Forest Feature Weighting Toggle
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        rf_weight_check = st.checkbox("🌲 Apply Random Forest Feature Weights", value=st.session_state.use_rf_weighting, help="Scales PCA feature space by Random Forest feature importance weights before clustering")
        if rf_weight_check != st.session_state.use_rf_weighting:
            st.session_state.use_rf_weighting = rf_weight_check
            st.session_state.clustering_result = None

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        c_act1, c_act2 = st.columns([1.2, 1.0])
        with c_act1:
            run_clust_btn = st.button("🚀 Start Clustering", type="primary", use_container_width=True)
        with c_act2:
            elbow_btn = st.button("📈 Find Optimal K", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with ctrl_col_right:
        st.markdown("""
        <div class="sf-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
                📦 Select Image Dataset (Upload 10-50 Images or Sample Batch)
            </div>
        """, unsafe_allow_html=True)

        data_source_mode = st.radio(
            "Image Input Mode",
            ["⚡ Quick-Load Fashion Sample Batch (Instant 1-Click)", "📤 Upload Multiple Files (10-50 Images)"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if data_source_mode.startswith("⚡"):
            st.markdown("""
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px;">
                Instantly load real fashion items across mixed categories (Tops, Trousers, Dresses, Shoes, Bags) directly from the Fashion-MNIST dataset:
            </p>
            """, unsafe_allow_html=True)

            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                if st.button("Load 15 Diverse Items", use_container_width=True):
                    indices = [i * 18 for i in range(15)]
                    imgs = [A["test_img"][i].reshape(28, 28) for i in indices]
                    st.session_state.cluster_images_cache = imgs
                    st.session_state.sample_batch_size = 15
                    st.session_state.clustering_result = None
                    st.rerun()
            with b_c2:
                if st.button("Load 30 Diverse Items", use_container_width=True):
                    indices = [i * 9 for i in range(30)]
                    imgs = [A["test_img"][i].reshape(28, 28) for i in indices]
                    st.session_state.cluster_images_cache = imgs
                    st.session_state.sample_batch_size = 30
                    st.session_state.clustering_result = None
                    st.rerun()
            with b_c3:
                if st.button("Load 50 Diverse Items", use_container_width=True):
                    indices = [i * 5 for i in range(50)]
                    imgs = [A["test_img"][i].reshape(28, 28) for i in indices]
                    st.session_state.cluster_images_cache = imgs
                    st.session_state.sample_batch_size = 50
                    st.session_state.clustering_result = None
                    st.rerun()

            if st.session_state.cluster_images_cache is None:
                indices = [i * 11 for i in range(25)]
                imgs = [A["test_img"][i].reshape(28, 28) for i in indices]
                st.session_state.cluster_images_cache = imgs

            st.success(f"✓ Active Batch: {len(st.session_state.cluster_images_cache)} balanced fashion images loaded in memory.")

        else:
            st.markdown("""
            <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px;">
                Select 10 to 50 apparel photos from your computer (JPG, PNG, WEBP):
            </p>
            """, unsafe_allow_html=True)

            multi_files = st.file_uploader(
                "Upload 10-50 images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="multi_img_uploader"
            )

            if multi_files:
                uploaded_pils = [Image.open(f) for f in multi_files]
                st.session_state.cluster_images_cache = uploaded_pils
                st.session_state.clustering_result = None
                st.success(f"✓ Uploaded {len(uploaded_pils)} custom images successfully!")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- CLUSTERING EXECUTION LOGIC WITH RANDOM FOREST INTEGRATION ---
    if run_clust_btn or (st.session_state.clustering_result is None and st.session_state.cluster_images_cache is not None):
        with st.spinner(f"Extracting features and running {st.session_state.selected_algo}..."):
            images_to_cluster = st.session_state.cluster_images_cache
            
            # Extract features with optional Random Forest weighting
            batch_data = ce.extract_features_batch(
                images_to_cluster,
                feature_extractor=feature_pipeline,
                rf_importances=rf_importances,
                use_rf_weighting=st.session_state.use_rf_weighting
            )
            feats = batch_data["features"]
            coords_2d = batch_data["coords_2d"]
            
            # Run selected algorithm
            if st.session_state.selected_algo == "K-Means":
                clust_res = ce.run_kmeans_clustering(feats, n_clusters=st.session_state.selected_k)
            elif st.session_state.selected_algo == "Hierarchical Clustering":
                clust_res = ce.run_hierarchical_clustering(feats, n_clusters=st.session_state.selected_k)
            else:
                clust_res = ce.run_dbscan_clustering(feats)
                
            # Run Random Forest Analysis on all batch images
            rf_item_results = ce.analyze_with_random_forest(batch_data["raw_784"], model, CLASSES, CLASS_ICONS)
            rf_cluster_insights = ce.compute_rf_cluster_insights(clust_res["labels"], rf_item_results, CLASSES, CLASS_ICONS)
            
            clust_res["coords_2d"] = coords_2d
            clust_res["batch_images"] = batch_data["processed_images"]
            clust_res["features"] = feats
            clust_res["var_ratio"] = batch_data["explained_variance_2d"]
            clust_res["rf_items"] = rf_item_results
            clust_res["rf_insights"] = rf_cluster_insights
            
            st.session_state.clustering_result = clust_res
            st.rerun()

    # --- OPTIMAL K (ELBOW METHOD) LOGIC ---
    if elbow_btn:
        with st.spinner("Calculating K-Means inertia across K=2 to K=8..."):
            if st.session_state.cluster_images_cache is not None:
                batch_data = ce.extract_features_batch(
                    st.session_state.cluster_images_cache,
                    feature_extractor=feature_pipeline,
                    rf_importances=rf_importances,
                    use_rf_weighting=st.session_state.use_rf_weighting
                )
                st.session_state.elbow_result = ce.compute_elbow_curve(batch_data["features"], min_k=2, max_k=8)

    # --- DISPLAY ELBOW METHOD IF TRIGGERED ---
    if st.session_state.elbow_result is not None:
        elb = st.session_state.elbow_result
        st.markdown(f"""
        <div class="sf-card" style="border: 1px solid rgba(56, 189, 248, 0.4); background: rgba(15, 23, 42, 0.85);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h3 style="font-size: 1.3rem; font-weight: 800; color: #ffffff; margin-bottom: 2px;">
                        Elbow Method for Optimal Cluster Selection
                    </h3>
                    <p style="font-size: 0.82rem; color: #94a3b8;">
                        Mathematical evaluation of K-Means WCSS (Inertia) vs. Number of Clusters (K)
                    </p>
                </div>
                <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); padding: 4px 14px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700; color: #38bdf8;">
                    Recommended: K = {elb['optimal_k']}
                </div>
            </div>
        """, unsafe_allow_html=True)

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elb["k_values"],
            y=elb["inertias"],
            mode="lines+markers",
            name="Inertia (WCSS)",
            line=dict(color="#818cf8", width=3),
            marker=dict(size=9, color="#38bdf8")
        ))
        opt_idx = elb["k_values"].index(elb["optimal_k"])
        fig_elbow.add_trace(go.Scatter(
            x=[elb["optimal_k"]],
            y=[elb["inertias"][opt_idx]],
            mode="markers",
            name="Elbow Knee Point",
            marker=dict(size=16, color="#ec4899", symbol="star")
        ))

        fig_elbow.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            margin=dict(l=40, r=40, t=20, b=40),
            height=300,
            xaxis=dict(title="Number of Clusters (K)", dtick=1, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Inertia (Sum of Squared Distances)", gridcolor="rgba(255,255,255,0.08)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.info(f"Recommendation: {elb['explanation']} Use the [{elb['optimal_k']}] cluster button to apply this recommended K.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 2D PCA VISUALIZATION & CLUSTER INSPECTION ---
    if st.session_state.clustering_result is not None:
        c_res = st.session_state.clustering_result
        coords = c_res["coords_2d"]
        labels = c_res["labels"]
        imgs = c_res["batch_images"]
        rf_items = c_res["rf_items"]
        rf_insights = c_res["rf_insights"]
        n_pts = len(coords)

        # Plotly DataFrame with Random Forest Predicted Categories
        df_plot = pd.DataFrame({
            "PCA Component 1": coords[:, 0],
            "PCA Component 2": coords[:, 1],
            "Cluster": [f"Cluster {l + 1}" if l != -1 else "Noise" for l in labels],
            "Cluster_ID": labels,
            "Image Index": [f"Item #{i+1}" for i in range(n_pts)],
            "RF Style": [f"{rf_items[i]['pred_icon']} {rf_items[i]['pred_name']} ({rf_items[i]['confidence']:.0%})" for i in range(n_pts)]
        })

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        viz_col_left, viz_col_right = st.columns([1.5, 1.0], gap="large")

        with viz_col_left:
            st.markdown(f"""
            <div class="sf-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">PCA &rarr; 2D Feature Space &rarr; Cluster Visualization</div>
                        <div style="font-size: 0.78rem; color: #94a3b8;">{"Scaled by Random Forest Importance Weights" if st.session_state.use_rf_weighting else "Standard 50-D Latent PCA Embeddings"}</div>
                    </div>
                    <span style="font-size: 0.75rem; color: #a855f7; background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.3); padding: 3px 10px; border-radius: 9999px;">
                        Variance: {c_res['var_ratio'][0]:.1%} + {c_res['var_ratio'][1]:.1%}
                    </span>
                </div>
            """, unsafe_allow_html=True)

            palette = ["#38bdf8", "#ec4899", "#10b981", "#a855f7", "#f59e0b", "#06b6d4", "#e11d48", "#84cc16"]
            fig_scatter = px.scatter(
                df_plot,
                x="PCA Component 1",
                y="PCA Component 2",
                color="Cluster",
                hover_data=["Image Index", "Cluster", "RF Style"],
                color_discrete_sequence=palette,
            )

            fig_scatter.update_traces(
                marker=dict(size=12, opacity=0.88, line=dict(width=1.5, color="#ffffff"))
            )

            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=420,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
                legend=dict(
                    bgcolor="rgba(15, 23, 42, 0.8)",
                    bordercolor="rgba(255, 255, 255, 0.1)",
                    borderwidth=1,
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Click or hover over individual points to inspect coordinates, cluster ID, and Random Forest predicted styles.")
            st.markdown("</div>", unsafe_allow_html=True)

        with viz_col_right:
            st.markdown("""
            <div class="sf-card" style="height: 100%;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
                    📊 Clustering Quality & RF Alignment
                </div>
            """, unsafe_allow_html=True)

            sil_val = c_res["silhouette_score"]
            db_val = c_res["davies_bouldin_score"]

            if sil_val > 0.4:
                sil_desc = "Strong Cluster Separation"
                sil_badge_color = "#10b981"
            elif sil_val > 0.2:
                sil_desc = "Moderate Visual Cohesion"
                sil_badge_color = "#38bdf8"
            else:
                sil_desc = "Overlapping / Low Separation"
                sil_badge_color = "#f59e0b"

            # Compute average Random Forest purity across all clusters
            purities = [v["purity_pct"] for k, v in rf_insights.items() if k != -1]
            avg_rf_purity = np.mean(purities) if purities else 0.0

            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">SILHOUETTE COEFFICIENT</span>
                    <span style="font-size: 0.72rem; color: {sil_badge_color}; background: rgba(16, 185, 129, 0.12); padding: 2px 8px; border-radius: 6px; font-weight: 700;">{sil_desc}</span>
                </div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 4px;">{sil_val:.3f}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">Measures how similar an image is to its own cluster compared to other clusters (-1 to +1).</div>
            </div>

            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">RANDOM FOREST CLUSTER PURITY</span>
                    <span style="font-size: 0.72rem; color: #34d399; background: rgba(16, 185, 129, 0.12); padding: 2px 8px; border-radius: 6px; font-weight: 700;">Tree Validation</span>
                </div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #34d399; margin-top: 4px;">{avg_rf_purity:.1f}%</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">Average percentage of cluster items agreeing on dominant Random Forest category.</div>
            </div>

            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px;">
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">EXECUTION LATENCY</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #ec4899; margin-top: 4px;">{c_res['execution_time_ms']:.1f} ms</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">Wall-clock time to converge algorithm on {n_pts} image vectors.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- CLUSTER IMAGE GALLERY CARDS WITH RANDOM FOREST LABELS ---
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h3 style="font-size: 1.45rem; font-weight: 800; color: #ffffff; margin-bottom: 2px;">
                    Cluster Image Groups &amp; Random Forest Style Analysis
                </h3>
                <p style="font-size: 0.82rem; color: #94a3b8;">
                    Explore visually grouped apparel items with dominant Random Forest style classifications
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        unique_clusters = np.unique(labels)
        unique_clusters = sorted(unique_clusters, key=lambda x: (x == -1, x))

        for c_id in unique_clusters:
            mask = (labels == c_id)
            c_indices = np.where(mask)[0]
            c_count = len(c_indices)
            c_name = f"Cluster {c_id + 1}" if c_id != -1 else "Noise / Outliers"
            c_color = palette[c_id % len(palette)] if c_id != -1 else "#64748b"
            
            c_info = rf_insights.get(c_id, {
                "dominant_style": "Unclassified",
                "dominant_icon": "👕",
                "purity_pct": 0.0,
                "avg_confidence_pct": 0.0
            })

            with st.expander(f"📦 {c_name} — {c_info['dominant_icon']} {c_info['dominant_style']} (Images: {c_count} | RF Purity: {c_info['purity_pct']}%)", expanded=True):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; background: rgba(30, 41, 59, 0.4); padding: 8px 14px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06);">
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <span style="background: {c_color}; color: #ffffff; padding: 3px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">
                            {c_name}
                        </span>
                        <span style="font-size: 0.85rem; color: #ffffff; font-weight: 700;">
                            Dominant Style: {c_info['dominant_icon']} {c_info['dominant_style']}
                        </span>
                    </div>
                    <div style="display: flex; gap: 14px; font-size: 0.78rem;">
                        <span style="color: #34d399; font-weight: 600;">Cluster Purity: {c_info['purity_pct']}%</span>
                        <span style="color: #38bdf8; font-weight: 600;">Avg RF Confidence: {c_info['avg_confidence_pct']}%</span>
                        <span style="color: #cbd5e1;">Total Items: {c_count}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Thumbnail Grid with Random Forest Tags
                grid_cols = st.columns(min(8, max(4, c_count)))
                for col_idx, item_idx in enumerate(c_indices):
                    target_col = grid_cols[col_idx % len(grid_cols)]
                    img_arr = imgs[item_idx]
                    rf_item = rf_items[item_idx]
                    with target_col:
                        st.image(img_arr.astype("uint8"), width=80)
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 2px;">
                            <div style="font-size: 0.72rem; font-weight: 700; color: #ffffff;">{rf_item['pred_icon']} {rf_item['pred_name']}</div>
                            <div style="font-size: 0.65rem; color: #38bdf8;">{rf_item['confidence']:.0%} RF Conf</div>
                        </div>
                        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: ALGORITHM COMPARISON & ENSEMBLE BENCHMARKS
# ==============================================================================
elif st.session_state.active_nav == "Algorithm Comparison":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(236, 72, 153, 0.15); border: 1px solid rgba(236, 72, 153, 0.35); color: #f472b6; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px;">
            <span>⚖️</span> Machine Learning Benchmark Suite
        </div>
        <h2 style="font-size: 2.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 4px;">Algorithm Comparison</h2>
        <p style="font-size: 0.95rem; color: #94a3b8;">
            Compare Supervised Random Forest Classifier with Unsupervised Clustering (K-Means, Hierarchical, DBSCAN).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Prepare batch features for real benchmarking
    if st.session_state.cluster_images_cache is None:
        indices = [i * 11 for i in range(30)]
        st.session_state.cluster_images_cache = [A["test_img"][i].reshape(28, 28) for i in indices]

    batch_data = ce.extract_features_batch(
        st.session_state.cluster_images_cache,
        feature_extractor=feature_pipeline,
        rf_importances=rf_importances,
        use_rf_weighting=st.session_state.use_rf_weighting
    )
    bm_results = ce.benchmark_all_clustering(batch_data["features"], n_clusters=st.session_state.selected_k)
    df_bm = bm_results["table"]

    st.markdown("""
    <div class="sf-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
            📊 Unsupervised Clustering Benchmarks on Feature Space
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_bm, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Supervised Random Forest vs Unsupervised Clustering Comparison Strip
    st.markdown("""
    <div class="sf-card" style="border: 1px solid rgba(16, 185, 129, 0.35);">
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;">
            🤖 Supervised Random Forest vs. Unsupervised K-Means Paradigm Comparison
        </div>
        <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 16px;">
            How the supervised ensemble classifier and unsupervised cluster partitioner work together in the StyleFind AI pipeline:
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="background: rgba(30, 41, 59, 0.5); padding: 16px; border-radius: 14px; border: 1px solid rgba(16, 185, 129, 0.2);">
                <div style="font-size: 0.95rem; font-weight: 700; color: #34d399; margin-bottom: 6px;">🌲 Random Forest Classifier</div>
                <div style="font-size: 0.78rem; color: #cbd5e1; line-height: 1.5;">
                    <p>• <strong>Paradigm:</strong> Supervised Learning (uses 60,000 ground-truth labels)</p>
                    <p>• <strong>Performance:</strong> 84.8% Test Accuracy, 0.847 Macro F1 Score</p>
                    <p>• <strong>Objective:</strong> Minimizes Gini Impurity to map apparel images into 10 known classes</p>
                    <p>• <strong>Role:</strong> Classifies uploaded apparel & computes per-item style confidence</p>
                </div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.5); padding: 16px; border-radius: 14px; border: 1px solid rgba(56, 189, 248, 0.2);">
                <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;">✨ K-Means Clustering</div>
                <div style="font-size: 0.78rem; color: #cbd5e1; line-height: 1.5;">
                    <p>• <strong>Paradigm:</strong> Unsupervised Learning (operates with zero labels)</p>
                    <p>• <strong>Performance:</strong> Evaluated via Silhouette Score & Davies-Bouldin Index</p>
                    <p>• <strong>Objective:</strong> Minimizes Within-Cluster Sum of Squares (Inertia)</p>
                    <p>• <strong>Role:</strong> Discovers organic visual silhouettes & groups unlabelled batches</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Comparison Charts
    c_chart1, c_chart2 = st.columns(2, gap="medium")
    with c_chart1:
        st.markdown("""
        <div class="sf-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 10px;">Silhouette Score (Higher is Better)</div>
        """, unsafe_allow_html=True)
        fig_sil = px.bar(
            df_bm,
            x="Algorithm",
            y="Silhouette Score",
            color="Algorithm",
            color_discrete_sequence=["#38bdf8", "#818cf8", "#ec4899"],
            text_auto=".3f"
        )
        fig_sil.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_chart2:
        st.markdown("""
        <div class="sf-card">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 10px;">Execution Time in Milliseconds (Faster is Better)</div>
        """, unsafe_allow_html=True)
        fig_time = px.bar(
            df_bm,
            x="Algorithm",
            y="Execution Time (ms)",
            color="Algorithm",
            color_discrete_sequence=["#34d399", "#f59e0b", "#f43f5e"],
            text_auto=".1f"
        )
        fig_time.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- ABOUT THE ALGORITHM SECTION ---
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 1.4rem; font-weight: 800; color: #ffffff; margin-bottom: 2px;">
            📖 About the Algorithms (Presentation &amp; Viva Reference)
        </h3>
        <p style="font-size: 0.82rem; color: #94a3b8;">
            Key theoretical foundations and algorithmic differences
        </p>
    </div>
    """, unsafe_allow_html=True)

    algo_c1, algo_c2, algo_c3 = st.columns(3, gap="medium")

    with algo_c1:
        st.markdown("""
        <div class="sf-card" style="border: 1px solid rgba(56, 189, 248, 0.4); height: 100%;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">Algorithm 1</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-bottom: 12px;">K-Means Clustering</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                <p><strong>Type:</strong> Unsupervised Learning</p>
                <p><strong>Purpose:</strong> Groups similar images based on feature vector distance.</p>
                <p><strong>Input:</strong> Image feature vectors (PCA 50-D)</p>
                <p><strong>Output:</strong> Discrete cluster assignments (0 to K-1)</p>
                <p><strong>Parameter:</strong> Number of clusters (K)</p>
                <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px;">
                    Iteratively computes cluster centroids to minimize within-cluster variance (Inertia).
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with algo_c2:
        st.markdown("""
        <div class="sf-card" style="border: 1px solid rgba(168, 85, 247, 0.4); height: 100%;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #c084fc; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">Algorithm 2</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-bottom: 12px;">Hierarchical Clustering</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                <p><strong>Type:</strong> Unsupervised Agglomerative</p>
                <p><strong>Purpose:</strong> Builds a hierarchy tree (dendrogram) of clusters.</p>
                <p><strong>Input:</strong> Pairwise image feature distances</p>
                <p><strong>Output:</strong> Hierarchical grouping</p>
                <p><strong>Parameter:</strong> Distance threshold or target K</p>
                <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px;">
                    Starts with every image as its own cluster and repeatedly merges closest pairs (Ward linkage).
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with algo_c3:
        st.markdown("""
        <div class="sf-card" style="border: 1px solid rgba(236, 72, 153, 0.4); height: 100%;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #f472b6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">Algorithm 3</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-bottom: 12px;">DBSCAN Clustering</div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                <p><strong>Type:</strong> Density-Based Spatial Clustering</p>
                <p><strong>Purpose:</strong> Discovers arbitrary-shaped clusters and filters outliers.</p>
                <p><strong>Input:</strong> Feature vectors with spatial density</p>
                <p><strong>Output:</strong> Clusters + Noise labels (-1)</p>
                <p><strong>Parameter:</strong> Neighborhood radius (Epsilon), MinPts</p>
                <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px;">
                    Does not require predefined K; marks low-density points as noise anomalies.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- REAL-WORLD INDUSTRY APPLICATIONS ---
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 1.4rem; font-weight: 800; color: #ffffff; margin-bottom: 2px;">
            🌐 Real-World Industry Applications
        </h3>
        <p style="font-size: 0.82rem; color: #94a3b8;">
            How image clustering and Random Forest classification power modern retail systems
        </p>
    </div>
    """, unsafe_allow_html=True)

    app_cols_1 = st.columns(3)
    apps = [
        ("👗 Fashion Image Organization", "Automated tagging and visual taxonomy generation across millions of user-uploaded catalogue items without manual curation."),
        ("🛍️ E-commerce Product Grouping", "Grouping identical or stylistically synonymous products from thousands of merchants to enhance catalogue browsing."),
        ("🔍 Visual Search", "Using latent feature space embeddings to allow customers to snap a photo and instantly find similar inventory items."),
        ("📊 Customer Preference Analysis", "Discovering visual style personas and trending silhouette clusters by clustering user interaction and purchase histories."),
        ("🗂️ Image Dataset Organization", "Structuring massive unlabelled visual datasets, detecting duplicate photos, and filtering corrupt uploads."),
        ("💡 Recommendation Systems", "Powering visual 'Complete the Look' feeds by recommending items from complementary aesthetic clusters.")
    ]

    for i in range(3):
        with app_cols_1[i]:
            title, desc = apps[i]
            st.markdown(f"""
            <div class="sf-card" style="height: 100%;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    app_cols_2 = st.columns(3)
    for i in range(3, 6):
        with app_cols_2[i - 3]:
            title, desc = apps[i]
            st.markdown(f"""
            <div class="sf-card" style="height: 100%;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">{title}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# --- Professional Footer ---
st.markdown("<div style='margin-top: 48px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 24px;'></div>", unsafe_allow_html=True)
foot_col1, foot_col2 = st.columns([2, 1])

with foot_col1:
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">✨👕 StyleFind AI</div>
    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
        AI-Powered Image Clustering & Style Prediction &middot; Random Forest & K-Means Architecture
    </div>
    """, unsafe_allow_html=True)

with foot_col2:
    st.markdown("""
    <div style="text-align: right; font-size: 0.75rem; color: #64748b;">
        Scikit-Learn &bull; Streamlit &bull; Plotly<br>
        Port 8501 &bull; Status: <span style="color: #10b981;">Active</span>
    </div>
    """, unsafe_allow_html=True)
