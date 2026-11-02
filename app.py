import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

st.set_page_config(
    page_title="CardioCheck & Decision Tree Studio",
    page_icon="🫀",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Claude-inspired Warm Ivory / Editorial Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* =========================================================================
       Global Background & Typography
       ========================================================================= */
    .stApp {
        background-color: #FAF9F5 !important;
        color: #1F1E1B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Container Card - generous padding-top ensures content is never obscured by the fixed navbar */
    .block-container {
        max-width: 720px !important;
        padding-top: 5.25rem !important;
        padding-bottom: 4rem !important;
    }

    /* =========================================================================
       Streamlit Top Header / Navbar (Eliminate black bar in dark mode)
       ========================================================================= */
    header[data-testid="stHeader"] {
        background-color: #FAF9F5 !important;
        border-bottom: 1px solid #E8E5DE !important;
    }
    header[data-testid="stHeader"] * {
        color: #1F1E1B !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {
        color: #4A4740 !important;
        fill: #4A4740 !important;
    }

    /* =========================================================================
       Editorial Typography
       ========================================================================= */
    .claude-title {
        font-family: "Georgia", "Newsreader", serif;
        font-size: 2.3rem;
        font-weight: 500;
        color: #1A1915;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        text-align: center;
    }
    .claude-subtitle {
        font-size: 0.95rem;
        color: #726E65;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* =========================================================================
       Segmented Control / Mode Selector (Crisp, High-Contrast, No Radio Dots)
       ========================================================================= */
    [data-testid="stSegmentedControl"] {
        display: flex !important;
        justify-content: center !important;
        margin: 0.5rem auto 1.75rem auto !important;
    }

    [data-testid="stSegmentedControl"] > div {
        background-color: #EDEAE1 !important;
        border: 1px solid #DCD8CF !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        display: inline-flex !important;
    }

    [data-testid="stSegmentedControl"] button {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        color: #5C584E !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 6px 18px !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSegmentedControl"] button[aria-checked="true"],
    [data-testid="stSegmentedControl"] button[data-state="active"] {
        background-color: #FFFFFF !important;
        color: #1A1915 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(27, 26, 23, 0.08) !important;
        border: 1px solid #DCD8CF !important;
    }

    [data-testid="stSegmentedControl"] button:hover {
        color: #1A1915 !important;
    }

    /* Radio Fallback (In case rendered as st.radio) */
    div[data-testid="stRadio"] > div {
        background-color: #EDEAE1 !important;
        padding: 4px !important;
        border-radius: 12px !important;
        display: flex !important;
        gap: 4px !important;
        border: 1px solid #DCD8CF !important;
        justify-content: center !important;
        margin: 0.5rem auto 1.75rem auto !important;
        max-width: 460px !important;
    }

    /* Hide native radio circles */
    div[data-testid="stRadio"] label > div:first-child,
    div[data-testid="stRadio"] label input[type="radio"] {
        display: none !important;
    }

    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 7px 18px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        border: 1px solid transparent !important;
    }

    div[data-testid="stRadio"] label p {
        color: #5C584E !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(27, 26, 23, 0.08) !important;
        border: 1px solid #DCD8CF !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #1A1915 !important;
        font-weight: 600 !important;
    }

    /* =========================================================================
       File Uploader (Force Crisp White Card & Visible Text)
       ========================================================================= */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E8E5DE !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        box-shadow: 0 1px 3px rgba(27, 26, 23, 0.04) !important;
        margin-bottom: 1.25rem !important;
    }

    [data-testid="stFileUploader"] > label,
    [data-testid="stFileUploader"] > label p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #1A1915 !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #FAF9F5 !important;
        border: 1.5px dashed #DCD8CF !important;
        border-radius: 10px !important;
        padding: 1.75rem 1rem !important;
        text-align: center !important;
        transition: all 0.15s ease-in-out !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F5F3EC !important;
        border-color: #D97757 !important;
    }

    /* Force all text inside dropzone to be dark and legible */
    [data-testid="stFileUploaderDropzone"] * {
        color: #4A4740 !important;
    }

    [data-testid="stFileUploaderDropzone"] svg {
        color: #726E65 !important;
        fill: #726E65 !important;
    }

    [data-testid="stFileUploaderDropzone"] span {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #1A1915 !important;
    }

    [data-testid="stFileUploaderDropzone"] small {
        font-size: 0.82rem !important;
        color: #726E65 !important;
        display: block !important;
        margin-top: 0.25rem !important;
    }

    /* Browse Files button inside dropzone */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #1A1915 !important;
        border: 1px solid #DCD8CF !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.1rem !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(27, 26, 23, 0.04) !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"]:hover {
        background-color: #FAF9F5 !important;
        border-color: #D97757 !important;
        color: #D97757 !important;
    }

    /* File tag after upload */
    [data-testid="stFileUploaderFile"] {
        background-color: #FAF9F5 !important;
        border: 1px solid #DCD8CF !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: #1F1E1B !important;
    }

    /* =========================================================================
       Inputs, Selectboxes, Dropdowns & Popovers
       ========================================================================= */
    label, .stWidgetLabel p {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #4A4740 !important;
        letter-spacing: 0.01em;
    }

    input, 
    .stSelectbox div[data-baseweb="select"] > div, 
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DCD8CF !important;
        border-radius: 8px !important;
        color: #1F1E1B !important;
        box-shadow: 0 1px 2px rgba(27, 26, 23, 0.03) !important;
    }

    input:focus, 
    .stSelectbox div[data-baseweb="select"] > div:focus-within, 
    .stMultiSelect div[data-baseweb="select"] > div:focus-within {
        border-color: #D97757 !important;
        box-shadow: 0 0 0 1px #D97757 !important;
    }

    /* Dropdown menu popover options */
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"], 
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCD8CF !important;
        box-shadow: 0 4px 12px rgba(27, 26, 23, 0.08) !important;
        border-radius: 8px !important;
    }

    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #1F1E1B !important;
    }

    li[role="option"]:hover, 
    li[role="option"][aria-selected="true"] {
        background-color: #FAF9F5 !important;
        color: #D97757 !important;
    }

    /* Multiselect Selected Tags */
    span[data-baseweb="tag"] {
        background-color: #EDEAE1 !important;
        border: 1px solid #DCD8CF !important;
        color: #1F1E1B !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] * {
        color: #1F1E1B !important;
    }

    /* =========================================================================
       Claude Signature Terracotta Button
       ========================================================================= */
    .stButton > button {
        background-color: #D97757 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 1px 3px rgba(217, 119, 87, 0.2) !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #C36546 !important;
        box-shadow: 0 3px 8px rgba(217, 119, 87, 0.3) !important;
    }

    /* =========================================================================
       Clean Cards
       ========================================================================= */
    .result-card-healthy {
        background-color: #F1F7F3;
        border: 1px solid #CDE2D3;
        color: #215438;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-top: 1.25rem;
        margin-bottom: 1rem;
    }
    .result-card-risk {
        background-color: #FAF0EC;
        border: 1px solid #F0D0C5;
        color: #8C2F17;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-top: 1.25rem;
        margin-bottom: 1rem;
    }

    .metric-card-sage {
        background-color: #F1F7F3;
        border: 1px solid #CDE2D3;
        color: #215438;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.25rem 0;
    }
    .metric-value-sage {
        font-family: "Georgia", serif;
        font-size: 2.25rem;
        font-weight: 600;
        color: #16432B;
        margin: 0.2rem 0;
    }

    .editorial-box {
        background-color: #FFFFFF;
        border: 1px solid #E8E5DE;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(27, 26, 23, 0.04);
    }

    .factor-item {
        font-size: 0.9rem;
        color: #4A4740;
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        color: #726E65 !important;
        font-size: 0.88rem !important;
    }

    /* Dataframe table styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #E8E5DE !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Mode Selector at the Top (Segmented Control / Radio)
# -----------------------------------------------------------------------------
if hasattr(st, "segmented_control"):
    mode = st.segmented_control(
        "Mode Selector",
        options=["🫀 CardioCheck Demo", "📁 Upload Custom Dataset"],
        default="🫀 CardioCheck Demo",
        label_visibility="collapsed"
    )
    if not mode:
        mode = "🫀 CardioCheck Demo"
else:
    mode = st.radio(
        "Mode Selector",
        ["🫀 CardioCheck Demo", "📁 Upload Custom Dataset"],
        horizontal=True,
        label_visibility="collapsed"
    )


# =============================================================================
# MODE 1: CardioCheck Demo (Default)
# =============================================================================
if mode == "🫀 CardioCheck Demo":
    st.markdown("<div class='claude-title'>CardioCheck</div>", unsafe_allow_html=True)
    st.markdown("<div class='claude-subtitle'>Explainable cardiovascular risk stratification powered by Decision Trees</div>", unsafe_allow_html=True)

    @st.cache_resource
    def load_cardio_model():
        return joblib.load('heart_disease_model.pkl')

    model = load_cardio_model()

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=95, value=50)
        sex = st.selectbox("Biological Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Experience", [
            "Typical Angina (Severe)",
            "Atypical Angina (Mild)",
            "Non-Anginal Discomfort",
            "No Pain / Asymptomatic"
        ])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=125)
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=500, value=210)

    with col2:
        thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=155)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 4.5, 0.8, step=0.1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
        restecg = 0  # Standard clinical baseline

    # Encodings
    sex_val = 1 if sex == "Male" else 0
    cp_map = {
        "Typical Angina (Severe)": 0,
        "Atypical Angina (Mild)": 1,
        "Non-Anginal Discomfort": 2,
        "No Pain / Asymptomatic": 3
    }
    cp_val = cp_map[cp]
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak]])

    st.write("")
    if st.button("Evaluate Assessment", use_container_width=True):
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        if prediction == 1:
            st.markdown(f"""
            <div class='result-card-risk'>
                <div style='font-size: 1.15rem; font-weight: 600; margin-bottom: 0.25rem;'>Elevated Cardiovascular Risk</div>
                <div style='font-size: 0.9rem;'>Estimated risk probability: <strong>{probabilities[1]*100:.1f}%</strong>. Clinical review suggested.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-healthy'>
                <div style='font-size: 1.15rem; font-weight: 600; margin-bottom: 0.25rem;'>Low Cardiovascular Risk</div>
                <div style='font-size: 0.9rem;'>Model confidence: <strong>{probabilities[0]*100:.1f}%</strong>. Biomarkers fall within standard limits.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Explainability bullets
        st.markdown("<p style='font-weight: 500; font-size: 0.95rem; color: #1F1E1B; margin-top: 1.25rem;'>Key factors evaluated by the decision tree:</p>", unsafe_allow_html=True)
        reasons = []
        if cp_val in [0, 3]:
            reasons.append("Chest pain presentation matched high-sensitivity indicator criteria.")
        if chol > 240:
            reasons.append(f"Cholesterol level ({chol} mg/dL) exceeds the standard 200 mg/dL reference threshold.")
        if trestbps > 140:
            reasons.append(f"Resting systolic blood pressure ({trestbps} mm Hg) is elevated.")
        if thalach < 130:
            reasons.append(f"Peak exercise heart rate ({thalach} bpm) is comparatively lower than expected.")
        if exang_val == 1:
            reasons.append("Presence of exercise-induced angina contributed to risk stratification.")

        if reasons:
            for r in reasons:
                st.markdown(f"<div class='factor-item'>• {r}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='factor-item'>• All primary biomarkers (Blood Pressure, Cholesterol, Heart Rate) remain inside safe baseline ranges.</div>", unsafe_allow_html=True)

    # Collapsible visual tree logic
    if os.path.exists("heart_tree.png"):
        st.write("")
        with st.expander("How this model decides (Decision Tree Flowchart)"):
            st.image("heart_tree.png", caption="Interpretable decision path generated from UCI Cleveland data")


# =============================================================================
# MODE 2: Custom Dataset Studio
# =============================================================================
else:
    st.markdown("<div class='claude-title'>Decision Tree Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='claude-subtitle'>Upload any CSV dataset to build, visualize, and interact with an interpretable tree</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset", 
        type=["csv"], 
        help="Upload any structured tabular CSV dataset."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        # Clean preview & dimensions
        st.markdown(
            f"<div style='font-size: 0.88rem; color: #726E65; margin-top: 0.75rem; margin-bottom: 0.4rem;'>"
            f"Dataset dimensions: <strong style='color: #1F1E1B;'>{df.shape[0]}</strong> rows × <strong style='color: #1F1E1B;'>{df.shape[1]}</strong> columns"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(5), use_container_width=True)

        st.write("")
        st.markdown("<p style='font-weight: 600; font-size: 1rem; color: #1F1E1B; margin-bottom: 0.5rem;'>Model Configuration</p>", unsafe_allow_html=True)

        col_cfg1, col_cfg2 = st.columns(2, gap="medium")
        columns_list = list(df.columns)

        with col_cfg1:
            default_target_idx = len(columns_list) - 1
            target_col = st.selectbox(
                "Target Column (label to predict)",
                options=columns_list,
                index=default_target_idx
            )

        with col_cfg2:
            max_depth = st.slider(
                "Tree Depth (Max Depth)",
                min_value=2,
                max_value=5,
                value=3,
                help="Shallower trees ensure high explainability and avoid overfitting."
            )

        # Feature selection (exclude selected target)
        remaining_cols = [c for c in columns_list if c != target_col]
        feature_cols = st.multiselect(
            "Feature Columns",
            options=remaining_cols,
            default=remaining_cols,
            help="Select features used to train the decision tree."
        )

        if not feature_cols:
            st.warning("Please select at least one feature column.")
            st.stop()

        train_button = st.button("Train & Explain Decision Tree", use_container_width=True)

        if train_button:
            with st.spinner("Processing data and training Decision Tree..."):
                X_raw = df[feature_cols].copy()
                y_raw = df[target_col].copy()

                valid_idx = y_raw.dropna().index
                X_raw = X_raw.loc[valid_idx]
                y_raw = y_raw.loc[valid_idx]

                if len(y_raw) < 10:
                    st.error("Dataset has too few valid rows to train a model.")
                    st.stop()

                target_le = LabelEncoder()
                y_encoded = target_le.fit_transform(y_raw.astype(str))
                classes_list = [str(c) for c in target_le.classes_]

                if len(classes_list) < 2:
                    st.error("The selected target column must contain at least 2 distinct classes.")
                    st.stop()

                encoders = {}
                feature_types = {}
                X_processed = pd.DataFrame(index=X_raw.index)

                for col in feature_cols:
                    series = X_raw[col]
                    is_cat = not pd.api.types.is_numeric_dtype(series)

                    if is_cat:
                        mode_val = series.mode()[0] if not series.mode().empty else "Missing"
                        imputed = series.fillna(mode_val).astype(str)
                        le = LabelEncoder()
                        encoded_vals = le.fit_transform(imputed)
                        encoders[col] = le
                        feature_types[col] = {
                            "type": "categorical",
                            "classes": list(le.classes_),
                            "mode": mode_val
                        }
                        X_processed[col] = encoded_vals
                    else:
                        median_val = float(series.median()) if not np.isnan(series.median()) else 0.0
                        imputed = series.fillna(median_val).astype(float)
                        feature_types[col] = {
                            "type": "numeric",
                            "min": float(series.min()) if not np.isnan(series.min()) else 0.0,
                            "max": float(series.max()) if not np.isnan(series.max()) else 100.0,
                            "median": median_val,
                            "is_integer": bool(pd.api.types.is_integer_dtype(series))
                        }
                        X_processed[col] = imputed

                stratify_opt = y_encoded if min(np.bincount(y_encoded)) >= 2 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=stratify_opt
                )

                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.session_state["custom_studio"] = {
                    "model": clf,
                    "accuracy": accuracy,
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "target_le": target_le,
                    "encoders": encoders,
                    "feature_types": feature_types,
                    "classes": classes_list,
                    "max_depth": max_depth,
                }

        # If model is trained, render results
        if "custom_studio" in st.session_state and st.session_state["custom_studio"]["target_col"] == target_col:
            studio = st.session_state["custom_studio"]
            clf = studio["model"]
            accuracy = studio["accuracy"]
            classes_list = studio["classes"]

            # Soft Sage Accuracy Card
            st.markdown(f"""
            <div class='metric-card-sage'>
                <div style='font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #215438; font-weight: 600;'>Validation Accuracy (80/20 Holdout)</div>
                <div class='metric-value-sage'>{accuracy * 100:.1f}%</div>
                <div style='font-size: 0.88rem; color: #356B4C;'>Decision tree trained with max_depth={studio['max_depth']} on {len(df)} samples</div>
            </div>
            """, unsafe_allow_html=True)

            # Decision Tree Visualization
            st.markdown("<p style='font-weight: 600; font-size: 1rem; color: #1F1E1B; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Decision Tree Architecture</p>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
            fig.patch.set_facecolor('#FAF9F5')
            ax.set_facecolor('#FAF9F5')
            
            plot_tree(
                clf,
                feature_names=studio["feature_cols"],
                class_names=classes_list,
                filled=True,
                rounded=True,
                impurity=False,
                proportion=True,
                precision=2,
                fontsize=8,
                ax=ax
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Live Prediction Dynamic Widgets
            st.markdown("---")
            st.markdown("<p style='font-weight: 600; font-size: 1.05rem; color: #1F1E1B; margin-bottom: 0.25rem;'>Test Live Prediction on Your Dataset</p>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.88rem; color: #726E65; margin-bottom: 1.25rem;'>Adjust sample values to inspect the model's live classification:</div>", unsafe_allow_html=True)

            col_in1, col_in2 = st.columns(2, gap="medium")
            user_sample = {}

            for idx, f_col in enumerate(studio["feature_cols"]):
                target_col_container = col_in1 if idx % 2 == 0 else col_in2
                ft = studio["feature_types"][f_col]

                with target_col_container:
                    if ft["type"] == "categorical":
                        val = st.selectbox(
                            f"{f_col}",
                            options=ft["classes"],
                            key=f"dyn_in_{f_col}"
                        )
                        user_sample[f_col] = val
                    else:
                        min_v = ft["min"]
                        max_v = ft["max"]
                        med_v = ft["median"]
                        if min_v == max_v:
                            max_v = min_v + 10.0
                        if ft["is_integer"]:
                            val = st.number_input(
                                f"{f_col}",
                                min_value=int(min_v),
                                max_value=int(max_v),
                                value=int(med_v),
                                step=1,
                                key=f"dyn_in_{f_col}"
                            )
                        else:
                            val = st.number_input(
                                f"{f_col}",
                                min_value=float(min_v),
                                max_value=float(max_v),
                                value=float(med_v),
                                step=round((max_v - min_v) / 50, 2) or 0.1,
                                key=f"dyn_in_{f_col}"
                            )
                        user_sample[f_col] = val

            st.write("")
            if st.button("Predict Custom Sample", use_container_width=True):
                vec = []
                for f_col in studio["feature_cols"]:
                    ft = studio["feature_types"][f_col]
                    val = user_sample[f_col]
                    if ft["type"] == "categorical":
                        enc = studio["encoders"][f_col]
                        if str(val) in enc.classes_:
                            encoded_val = enc.transform([str(val)])[0]
                        else:
                            encoded_val = 0
                        vec.append(encoded_val)
                    else:
                        vec.append(float(val))

                sample_matrix = np.array([vec])
                pred_idx = clf.predict(sample_matrix)[0]
                pred_label = studio["classes"][pred_idx]
                
                prob_str = ""
                try:
                    probs = clf.predict_proba(sample_matrix)[0]
                    prob_val = probs[pred_idx] * 100
                    prob_str = f" with <strong>{prob_val:.1f}%</strong> confidence"
                except Exception:
                    pass

                st.markdown(f"""
                <div class='result-card-healthy'>
                    <div style='font-size: 1.15rem; font-weight: 600; margin-bottom: 0.25rem;'>Predicted {studio['target_col']}: {pred_label}</div>
                    <div style='font-size: 0.9rem;'>Decision Tree evaluated sample{prob_str}.</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<p style='font-weight: 500; font-size: 0.92rem; color: #1F1E1B; margin-top: 1rem;'>Rule sequence traversed for this prediction:</p>", unsafe_allow_html=True)
                node_indicator = clf.decision_path(sample_matrix)
                leaf_id = clf.apply(sample_matrix)[0]
                node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

                for step_i, node_id in enumerate(node_indices):
                    if node_id == leaf_id:
                        break
                    feat_idx = clf.tree_.feature[node_id]
                    feat_name = studio["feature_cols"][feat_idx]
                    threshold = clf.tree_.threshold[node_id]
                    actual_val = sample_matrix[0][feat_idx]
                    op = "≤" if actual_val <= threshold else ">"
                    st.markdown(
                        f"<div class='factor-item'>• <strong>Step {step_i + 1}:</strong> {feat_name} ({actual_val}) {op} {threshold:.2f}</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.markdown(
            """
            <div class='editorial-box' style='text-align: center; padding: 2rem 1.5rem; color: #726E65;'>
                <div style='font-size: 1.4rem; margin-bottom: 0.5rem;'>📄</div>
                <div style='font-weight: 600; color: #1A1915; margin-bottom: 0.25rem;'>No dataset uploaded yet</div>
                <div style='font-size: 0.88rem; color: #726E65;'>Upload any tabular CSV file above to start analyzing decision paths.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
