"""
CardioCheck AI - Heart Disease Risk Classification & Explainability Web App
Powered by Decision Tree Classifier & Streamlit
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioCheck AI | Heart Disease Risk & Explainability",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "heart_disease_model.pkl"
TREE_IMAGE_PATH = "heart_tree.png"
DATASET_PATH = "heart.csv"

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]

CP_DESCRIPTIONS = {
    0: "0 - Typical Angina (Chest pain on exertion)",
    1: "1 - Atypical Angina (Discomfort not meeting angina criteria)",
    2: "2 - Non-Anginal Pain (Non-cardiac chest discomfort)",
    3: "3 - Asymptomatic (Silent ischemia / no pain warning)",
}

RESTECG_DESCRIPTIONS = {
    0: "0 - Normal",
    1: "1 - ST-T Wave Abnormality (Inversions / elevation)",
    2: "2 - Left Ventricular Hypertrophy (Estes' criteria)",
}

# -----------------------------------------------------------------------------
# Custom Medical UI Styling (CSS)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Modern Medical Typography & Body */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Banner */
.medical-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
    border-radius: 16px;
    padding: 24px 32px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.medical-header h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 12px;
}

.medical-header p {
    margin: 6px 0 0 0;
    font-size: 0.95rem;
    color: #94A3B8;
}

/* Badge styling */
.badge-tech {
    background-color: rgba(255, 255, 255, 0.12);
    color: #38BDF8;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-top: 10px;
    margin-right: 6px;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Outcome Cards */
.risk-card {
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}

.risk-card-low {
    background: linear-gradient(145deg, #ECFDF5 0%, #D1FAE5 100%);
    border: 2px solid #10B981;
    color: #064E3B;
}

.risk-card-high {
    background: linear-gradient(145deg, #FEF2F2 0%, #FEE2E2 100%);
    border: 2px solid #EF4444;
    color: #7F1D1D;
}

.risk-title {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.risk-subtitle {
    font-size: 0.95rem;
    margin: 0 0 16px 0;
    opacity: 0.9;
}

/* Metric Display Pills */
.metric-pill {
    background: white;
    padding: 10px 16px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    display: inline-flex;
    flex-direction: column;
    margin-right: 12px;
    margin-bottom: 8px;
    min-width: 140px;
}

.metric-pill-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748B;
}

.metric-pill-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0F172A;
}

/* Step-by-step Rule Cards */
.rule-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 5px solid #0284C7;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    border-top: 1px solid #F1F5F9;
    border-right: 1px solid #F1F5F9;
    border-bottom: 1px solid #F1F5F9;
}

.rule-card-decision {
    border-left-color: #EF4444;
    background: #FFFBFB;
}

.rule-card-healthy {
    border-left-color: #10B981;
    background: #F8FCF9;
}

.rule-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.rule-step-tag {
    background: #E2E8F0;
    color: #334155;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}

.rule-feature-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
}

.rule-condition {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.88rem;
    color: #0369A1;
    background: #F0F9FF;
    padding: 4px 8px;
    border-radius: 6px;
    display: inline-block;
    margin: 6px 0;
}

.rule-explanation {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.45;
    margin: 4px 0 0 0;
}

/* Leaf Card */
.leaf-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: white;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 16px;
    border: 1px solid #334155;
}

.leaf-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #38BDF8;
    margin: 0 0 6px 0;
}

.leaf-stats {
    font-size: 0.85rem;
    color: #CBD5E1;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model & Asset Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load pre-trained Decision Tree model from disk."""
    if not os.path.exists(MODEL_PATH):
        # Auto-train if not found
        import subprocess
        subprocess.run(["python3", "train_model.py"], check=True)
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    """Load reference CSV for statistics and distribution benchmarks."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


try:
    model = load_model()
    dataset_df = load_reference_data()
except Exception as e:
    st.error(f"Error initializing CardioCheck AI: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Decision Tree Rule Tracer & Clinical Interpreter
# -----------------------------------------------------------------------------
def get_clinical_feature_name(feat_key: str) -> str:
    mapping = {
        "age": "Patient Age",
        "sex": "Biological Sex",
        "cp": "Chest Pain Classification",
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "fbs": "Fasting Blood Sugar",
        "restecg": "Resting Electrocardiogram (ECG)",
        "thalach": "Maximum Achieved Heart Rate",
        "exang": "Exercise-Induced Angina",
        "oldpeak": "ST Depression (Exercise vs Rest)",
    }
    return mapping.get(feat_key, feat_key)


def interpret_split(feat_key: str, val: float, threshold: float, went_left: bool) -> str:
    """Generate meaningful clinical context for why this tree branch was traversed."""
    direction = "<=" if went_left else ">"

    if feat_key == "cp":
        if went_left:
            return (
                f"Patient presents with Typical/Atypical/Non-Anginal chest pain (type {int(val)} <= {threshold:.1f}). "
                "Traversed the branch evaluated for specific exertional ischemia markers."
            )
        else:
            return (
                f"Patient exhibits Asymptomatic / Silent Ischemic presentation (type {int(val)} > {threshold:.1f}). "
                "In cardiac stress protocols, silent ischemia strongly correlates with severe underlying coronary stenosis."
            )

    elif feat_key == "oldpeak":
        if went_left:
            return (
                f"ST segment depression of {val:.1f} mm is within the lower/moderate range (<= {threshold:.2f} mm), "
                "reflecting less acute myocardial stress under exertion."
            )
        else:
            return (
                f"ST segment depression of {val:.1f} mm exceeds critical clinical threshold (> {threshold:.2f} mm), "
                "a hallmark electrocardiographic sign of subendocardial ischemia."
            )

    elif feat_key == "thalach":
        if went_left:
            return (
                f"Max achieved heart rate of {int(val)} bpm is restricted (<= {threshold:.1f} bpm), "
                "indicating chronotropic incompetence and diminished cardiovascular exertion capacity."
            )
        else:
            return (
                f"Max achieved heart rate of {int(val)} bpm exceeds {threshold:.1f} bpm, "
                "demonstrating strong cardiovascular functional reserve."
            )

    elif feat_key == "trestbps":
        if went_left:
            return (
                f"Resting blood pressure of {int(val)} mm Hg is at or below {threshold:.1f} mm Hg, "
                "reducing vascular wall tension."
            )
        else:
            return (
                f"Resting blood pressure of {int(val)} mm Hg is elevated (> {threshold:.1f} mm Hg), "
                "contributing to arterial stiffness and hypertensive heart disease risk."
            )

    elif feat_key == "age":
        if went_left:
            return (
                f"Patient age of {int(val)} is <= {threshold:.1f} years, "
                "placing them in a lower chronological baseline risk cohort."
            )
        else:
            return (
                f"Patient age of {int(val)} exceeds {threshold:.1f} years, "
                "a demographic factor linked with cumulative vascular atherogenesis."
            )

    elif feat_key == "sex":
        if went_left:
            return (
                f"Female patient (sex=0 <= {threshold:.1f}). In this cohort, baseline coronary disease prevalence was lower."
            )
        else:
            return (
                f"Male patient (sex=1 > {threshold:.1f}). Male biology carries statistically elevated incidence of coronary events."
            )

    elif feat_key == "exang":
        if went_left:
            return f"No exercise-induced angina reported (<= {threshold:.1f}), ruling out active exertional pain."
        else:
            return f"Exercise-induced angina present (> {threshold:.1f}), signifying coronary insufficiency during physical stress."

    elif feat_key == "chol":
        if went_left:
            return f"Serum cholesterol ({int(val)} mg/dL <= {threshold:.1f} mg/dL) satisfies conservative bounds."
        else:
            return f"Serum cholesterol ({int(val)} mg/dL > {threshold:.1f} mg/dL) reflects dyslipidemia and plaque risk."

    return f"{get_clinical_feature_name(feat_key)} value ({val}) is {direction} {threshold:.2f} threshold."


def trace_decision_path(clf, patient_df: pd.DataFrame):
    """Trace exact decision tree path and return structured rule steps."""
    node_indicator = clf.decision_path(patient_df)
    leaf_id = clf.apply(patient_df)[0]
    node_indices = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    steps = []
    for i, node_id in enumerate(node_indices):
        if node_id == leaf_id:
            # Leaf node reached
            counts = clf.tree_.value[node_id][0]
            total = np.sum(counts)
            prob_healthy = counts[0] / total
            prob_disease = counts[1] / total
            leaf_info = {
                "node_id": node_id,
                "total_samples": int(total),
                "healthy_count": int(counts[0]),
                "disease_count": int(counts[1]),
                "prob_healthy": prob_healthy,
                "prob_disease": prob_disease,
            }
            break

        feat_idx = clf.tree_.feature[node_id]
        feat_name = clf.feature_names_in_[feat_idx]
        threshold = clf.tree_.threshold[node_id]
        val = patient_df[feat_name].iloc[0]

        left_child = clf.tree_.children_left[node_id]
        next_node = node_indices[i + 1]
        went_left = (next_node == left_child)

        explanation = interpret_split(feat_name, val, threshold, went_left)

        steps.append({
            "step_num": i + 1,
            "node_id": node_id,
            "feature": feat_name,
            "feature_label": get_clinical_feature_name(feat_name),
            "patient_value": val,
            "threshold": threshold,
            "went_left": went_left,
            "operator": "<=" if went_left else ">",
            "branch_direction": "LEFT (Branch Condition Met)" if went_left else "RIGHT (Branch Condition Exceeded)",
            "explanation": explanation,
        })

    return steps, leaf_info


# -----------------------------------------------------------------------------
# Header Presentation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="medical-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <h1>🫀 CardioCheck AI</h1>
                <p>Cardiovascular Risk Stratification & White-Box Decision Tree Explainability</p>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span class="badge-tech">SCIKIT-LEARN 1.9</span>
                <span class="badge-tech">DECISION TREE (DEPTH 4)</span>
                <span class="badge-tech">UCI CLEVELAND COHORT</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Presets and Patient Clinical Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📋 Patient Clinical Profile")
    st.caption("Adjust laboratory parameters, hemodynamics, and stress test readings.")

    # Preset Quick Loaders
    st.markdown("#### ⚡ Clinical Presets")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_choice = None
    if col_p1.button("Healthy", use_container_width=True, help="Load low-risk patient profile"):
        preset_choice = "healthy"
    if col_p2.button("Borderline", use_container_width=True, help="Load moderate/borderline profile"):
        preset_choice = "borderline"
    if col_p3.button("High Risk", use_container_width=True, help="Load high-risk patient profile"):
        preset_choice = "high_risk"

    # Default values dictionary
    if "patient_params" not in st.session_state:
        st.session_state.patient_params = {
            "age": 52,
            "sex": 1,
            "cp": 1,
            "trestbps": 125,
            "chol": 210,
            "fbs": 0,
            "restecg": 0,
            "thalach": 165,
            "exang": 0,
            "oldpeak": 0.4,
        }

    # Handle preset application
    if preset_choice == "healthy":
        st.session_state.patient_params = {
            "age": 42,
            "sex": 0,
            "cp": 1,
            "trestbps": 118,
            "chol": 195,
            "fbs": 0,
            "restecg": 0,
            "thalach": 172,
            "exang": 0,
            "oldpeak": 0.2,
        }
    elif preset_choice == "borderline":
        st.session_state.patient_params = {
            "age": 58,
            "sex": 1,
            "cp": 2,
            "trestbps": 138,
            "chol": 245,
            "fbs": 0,
            "restecg": 1,
            "thalach": 145,
            "exang": 0,
            "oldpeak": 1.2,
        }
    elif preset_choice == "high_risk":
        st.session_state.patient_params = {
            "age": 67,
            "sex": 1,
            "cp": 3,
            "trestbps": 160,
            "chol": 286,
            "fbs": 1,
            "restecg": 2,
            "thalach": 108,
            "exang": 1,
            "oldpeak": 2.6,
        }

    p = st.session_state.patient_params

    st.markdown("---")
    st.markdown("#### 1. Demographics & Symptoms")
    age = st.slider(
        "Age (years)",
        min_value=25,
        max_value=85,
        value=p["age"],
        help="Patient age in years.",
    )

    sex_str = st.radio(
        "Biological Sex",
        options=["Male", "Female"],
        index=0 if p["sex"] == 1 else 1,
        horizontal=True,
    )
    sex = 1 if sex_str == "Male" else 0

    cp_options = list(CP_DESCRIPTIONS.keys())
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=cp_options,
        index=cp_options.index(p["cp"]),
        format_func=lambda x: CP_DESCRIPTIONS[x],
        help="Type of chest discomfort reported by the patient.",
    )

    exang_str = st.radio(
        "Exercise-Induced Angina (exang)",
        options=["No (0)", "Yes (1)"],
        index=p["exang"],
        horizontal=True,
        help="Whether angina chest pain is provoked during physical exercise.",
    )
    exang = 1 if "Yes" in exang_str else 0

    st.markdown("---")
    st.markdown("#### 2. Hemodynamics & Biochemistry")
    trestbps = st.slider(
        "Resting Blood Pressure (trestbps in mm Hg)",
        min_value=90,
        max_value=205,
        value=p["trestbps"],
        step=1,
        help="Resting arterial blood pressure measured on hospital admission.",
    )

    chol = st.slider(
        "Serum Cholesterol (chol in mg/dL)",
        min_value=120,
        max_value=560,
        value=p["chol"],
        step=2,
        help="Total serum cholesterol level.",
    )

    fbs_str = st.radio(
        "Fasting Blood Sugar > 120 mg/dL (fbs)",
        options=["Normal / No (0)", "Elevated / Yes (1)"],
        index=p["fbs"],
        horizontal=True,
        help="Fasting blood sugar > 120 mg/dL indicates elevated diabetic/pre-diabetic risk.",
    )
    fbs = 1 if "Yes" in fbs_str else 0

    st.markdown("---")
    st.markdown("#### 3. Cardiac Stress & ECG Readings")
    thalach = st.slider(
        "Maximum Heart Rate Achieved (thalach in bpm)",
        min_value=70,
        max_value=210,
        value=p["thalach"],
        help="Peak heart rate attained during treadmill stress testing.",
    )

    oldpeak = st.slider(
        "ST Depression Induced by Exercise (oldpeak in mm)",
        min_value=0.0,
        max_value=6.2,
        value=float(p["oldpeak"]),
        step=0.1,
        help="ST depression in mm induced by exercise relative to resting baseline ECG.",
    )

    restecg_options = list(RESTECG_DESCRIPTIONS.keys())
    restecg = st.selectbox(
        "Resting ECG Results (restecg)",
        options=restecg_options,
        index=restecg_options.index(p["restecg"]),
        format_func=lambda x: RESTECG_DESCRIPTIONS[x],
        help="Resting electrocardiographic evaluation.",
    )

    # Update session state with current inputs
    st.session_state.patient_params = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
    }

# -----------------------------------------------------------------------------
# Prepare Patient Sample & Inference
# -----------------------------------------------------------------------------
sample_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
}])[FEATURE_NAMES]

# Run prediction
pred_class = model.predict(sample_df)[0]
probs = model.predict_proba(sample_df)[0]
prob_healthy = probs[0]
prob_disease = probs[1]
is_high_risk = pred_class == 1

# Trace explainable tree path
steps, leaf_info = trace_decision_path(model, sample_df)


# -----------------------------------------------------------------------------
# Main Dashboard UI
# -----------------------------------------------------------------------------
tab_analysis, tab_tree_viz, tab_model_metrics, tab_dataset = st.tabs([
    "🩺 Patient Risk & Explainability",
    "🌳 Visual Decision Tree Diagram",
    "📊 Model Performance & Metrics",
    "📁 Dataset Explorer",
])

# =============================================================================
# TAB 1: Patient Risk & Explainability
# =============================================================================
with tab_analysis:
    col_res, col_vitals = st.columns([1.1, 0.9])

    with col_res:
        if is_high_risk:
            st.markdown(
                f"""
                <div class="risk-card risk-card-high">
                    <div class="risk-title">
                        <span>🚨</span> HIGH RISK DETECTED
                    </div>
                    <div class="risk-subtitle">
                        Cardiovascular profile indicates significant evidence of Coronary Artery Disease.
                    </div>
                    <div style="margin-top: 14px;">
                        <div class="metric-pill">
                            <span class="metric-pill-label">Risk Probability</span>
                            <span class="metric-pill-value" style="color: #DC2626;">{prob_disease * 100:.1f}%</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Healthy Probability</span>
                            <span class="metric-pill-value" style="color: #475569;">{prob_healthy * 100:.1f}%</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Clinical Recommendation</span>
                            <span class="metric-pill-value" style="font-size: 0.95rem; color: #DC2626;">Urgent Cardiology Referral</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="risk-card risk-card-low">
                    <div class="risk-title">
                        <span>✅</span> LOW RISK (HEALTHY PROFILE)
                    </div>
                    <div class="risk-subtitle">
                        Cardiovascular parameters fall within expected non-pathological thresholds.
                    </div>
                    <div style="margin-top: 14px;">
                        <div class="metric-pill">
                            <span class="metric-pill-label">Healthy Probability</span>
                            <span class="metric-pill-value" style="color: #059669;">{prob_healthy * 100:.1f}%</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Disease Probability</span>
                            <span class="metric-pill-value" style="color: #64748B;">{prob_disease * 100:.1f}%</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-pill-label">Clinical Recommendation</span>
                            <span class="metric-pill-value" style="font-size: 0.95rem; color: #059669;">Routine Annual Checkup</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Risk Probability Progress Bar
        st.markdown(f"**Cardiac Disease Risk Spectrum: {prob_disease * 100:.1f}%**")
        st.progress(float(prob_disease))

    with col_vitals:
        st.markdown("#### 🔬 Patient Diagnostic Summary")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.metric(
                label="Resting Blood Pressure",
                value=f"{trestbps} mm Hg",
                delta="Elevated (>130)" if trestbps > 130 else "Normal",
                delta_color="inverse",
            )
            st.metric(
                label="Peak Exertion Heart Rate",
                value=f"{thalach} bpm",
                delta="Reduced (<140)" if thalach < 140 else "Optimal Reserve",
                delta_color="normal",
            )
        with vcol2:
            st.metric(
                label="Serum Cholesterol",
                value=f"{chol} mg/dL",
                delta="Borderline/High (>200)" if chol > 200 else "Optimal",
                delta_color="inverse",
            )
            st.metric(
                label="ST Depression (oldpeak)",
                value=f"{oldpeak:.1f} mm",
                delta="Ischemic Sign (>=1.0)" if oldpeak >= 1.0 else "Normal (<1.0)",
                delta_color="inverse",
            )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # "Why did AI predict this?" - Dynamic Decision Tree Explainability
    # -------------------------------------------------------------------------
    st.markdown("### 💡 Why did AI predict this? (Dynamic Branch Explainability)")
    st.markdown(
        """
        Unlike black-box models, a **Decision Tree (depth 4)** provides exact, 100% transparent audit trails.
        Below is the precise physiological rule pathway traversed through the decision tree from the root node to the terminal diagnosis:
        """
    )

    card_class = "rule-card-decision" if is_high_risk else "rule-card-healthy"

    for step in steps:
        direction_icon = "⬅️" if step["went_left"] else "➡️"
        branch_color = "#0284C7" if step["went_left"] else "#EA580C"

        st.markdown(
            f"""
            <div class="rule-card {card_class}">
                <div class="rule-header">
                    <span class="rule-feature-title">{step['feature_label']}</span>
                    <span class="rule-step-tag">Step {step['step_num']} • Node #{step['node_id']}</span>
                </div>
                <div>
                    <span class="rule-condition">
                        Patient {step['feature']} = <b>{step['patient_value']}</b> {step['operator']} Threshold <b>{step['threshold']:.2f}</b>
                    </span>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {branch_color}; margin-left: 10px;">
                        {direction_icon} {step['branch_direction']}
                    </span>
                </div>
                <p class="rule-explanation">
                    {step['explanation']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Terminal Leaf Node Summary Card
    leaf_pred_label = "Heart Disease (High Risk)" if is_high_risk else "Healthy (Low Risk)"
    leaf_pred_color = "#F87171" if is_high_risk else "#34D399"
    st.markdown(
        f"""
        <div class="leaf-card">
            <div class="leaf-title">🏁 Terminal Leaf Decision: Node #{leaf_info['node_id']}</div>
            <div class="leaf-stats">
                • <b>Diagnosis:</b> <span style="color: {leaf_pred_color}; font-weight: 700;">{leaf_pred_label}</span><br>
                • <b>Cohort Calibration:</b> {leaf_info['total_samples']} training patients reached this exact node.<br>
                • <b>Historical Cohort Breakdown:</b> {leaf_info['healthy_count']} Healthy ({leaf_info['prob_healthy'] * 100:.1f}%) | 
                {leaf_info['disease_count']} Heart Disease ({leaf_info['prob_disease'] * 100:.1f}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 2: Visual Decision Tree Diagram
# =============================================================================
with tab_tree_viz:
    st.markdown("### 🌳 Complete Decision Tree Architecture (High Resolution)")
    st.markdown(
        """
        This high-resolution diagram maps every decision boundary learned by the model on the UCI Cleveland dataset.
        Nodes are color-coded: **Orange / Red** denotes heart disease predisposition, while **Blue** denotes healthy profiles.
        Color saturation corresponds to class certainty (lower Gini impurity).
        """
    )

    if os.path.exists(TREE_IMAGE_PATH):
        st.image(TREE_IMAGE_PATH, caption="DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)", use_container_width=True)

        with open(TREE_IMAGE_PATH, "rb") as f:
            st.download_button(
                label="📥 Download High-Resolution Diagram (heart_tree.png)",
                data=f,
                file_name="heart_tree.png",
                mime="image/png",
            )
    else:
        st.warning("Decision tree diagram not found. Run `python train_model.py` to generate it.")


# =============================================================================
# TAB 3: Model Performance & Metrics
# =============================================================================
with tab_model_metrics:
    st.markdown("### 📊 Model Architecture & Validation Performance")
    st.markdown(
        """
        The classifier was trained on an 80/20 stratified split of the authentic UCI Cleveland Clinic Heart Disease dataset.
        A constrained depth of `max_depth=4` was selected to prevent overfitting while preserving full clinical interpretability.
        """
    )

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Test Accuracy", "73.33%", help="Overall classification accuracy on unseen test cohort")
    mcol2.metric("Disease Precision", "73.08%", help="Precision for Heart Disease class")
    mcol3.metric("Disease Recall", "67.86%", help="Sensitivity / True Positive Rate for Heart Disease")
    mcol4.metric("Macro F1-Score", "0.73", help="Harmonic mean of precision and recall")

    st.markdown("#### Feature Importances (Gini Importance)")
    importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
    st.bar_chart(importances)

    st.markdown("#### Model Specification")
    st.code(
        """
DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)
        """,
        language="python",
    )


# =============================================================================
# TAB 4: Dataset Explorer
# =============================================================================
with tab_dataset:
    st.markdown("### 📁 UCI Heart Disease Dataset Explorer")
    if dataset_df is not None:
        st.markdown(f"**Total Records:** `{len(dataset_df)}` | **Total Features:** `{len(dataset_df.columns)}`")
        st.dataframe(dataset_df.head(25), use_container_width=True)

        st.markdown("#### Feature Schema Description")
        schema_data = {
            "Feature": FEATURE_NAMES + ["target"],
            "Description": [
                "Age of patient in years",
                "Biological Sex (1 = male, 0 = female)",
                "Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)",
                "Resting blood pressure in mm Hg on admission",
                "Serum cholesterol in mg/dL",
                "Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)",
                "Resting ECG (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)",
                "Maximum heart rate achieved during exercise stress",
                "Exercise induced angina (1 = yes, 0 = no)",
                "ST depression induced by exercise relative to rest (in mm)",
                "Diagnosis target (0 = Healthy, 1 = Heart Disease)",
            ],
            "Type": ["Integer", "Binary", "Categorical (0-3)", "Integer", "Integer", "Binary", "Categorical (0-2)", "Integer", "Binary", "Float", "Binary Target"],
        }
        st.table(pd.DataFrame(schema_data))
    else:
        st.info("Dataset heart.csv is loading or not found.")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.82rem; padding: 12px 0;">
        ⚠️ <b>Disclaimer:</b> CardioCheck AI is an educational clinical decision support demonstration built on the UCI Heart Disease cohort.
        It should not be utilized as a substitute for professional medical advice, clinical diagnosis, or patient management.
    </div>
    """,
    unsafe_allow_html=True,
)

