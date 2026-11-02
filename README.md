# 🫀 CardioCheck AI: Heart Disease Risk Classification & Explainability

A production-ready Machine Learning system and modern interactive medical dashboard for cardiovascular risk prediction using an interpretable **Decision Tree Classifier (depth=4)** and **Streamlit**.

---

## 🌟 Key Features

- **White-Box Explainability**: Real-time extraction of decision paths via `model.decision_path()`, explaining every physiological threshold and branch traversed from root to leaf node.
- **Interpretable ML**: Constrained `DecisionTreeClassifier(max_depth=4, random_state=42)` preventing overfitting and enabling direct clinical verification.
- **Modern Medical Dashboard**: Custom clinical CSS theme with responsive risk indicators, metric pills, progress bars, interactive sliders, and quick clinical preset buttons (*Healthy*, *Borderline*, *High Risk*).
- **High-Resolution Architecture Diagram**: Automatically rendered 300 DPI visualization (`heart_tree.png`) available for inspection and download directly in the UI.
- **Authentic UCI Cleveland Schema**: 10 diagnostic features + 1 target (`0 = Healthy`, `1 = Heart Disease`).

---

## 📂 Project Architecture

```
.
├── heart.csv               # Cleaned UCI Heart Disease dataset (10 features + target)
├── data_loader.py          # Data ingestion script (authentic UCI fetch + clinical fallback)
├── train_model.py          # Decision Tree training, evaluation & tree visualization generator
├── heart_disease_model.pkl # Serialized trained DecisionTreeClassifier artifact
├── heart_tree.png          # High-resolution (300 DPI) decision tree diagram
├── app.py                  # Modern Streamlit medical UI with dynamic explainability
├── requirements.txt        # Production dependencies
└── README.md               # Documentation and quick start guide
```

---

## ⚙️ Installation & Setup

### 1. Clone / Navigate to Workspace
```bash
cd /home/abhay/vscode/College/ML
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Train the Decision Tree Model
Trains the model, outputs performance metrics, and generates `heart_disease_model.pkl` and `heart_tree.png`:
```bash
python3 train_model.py
```

#### Sample Evaluation Output:
- **Test Accuracy**: `73.33%`
- **Disease Precision**: `73.08%`
- **Disease Recall**: `67.86%`
- **Confusion Matrix**:
  - True Negatives (Healthy): 25
  - True Positives (Disease): 19
- **Top Feature Importances**: Chest Pain Type (`cp`), ST Depression (`oldpeak`), Peak Heart Rate (`thalach`), Biological Sex (`sex`), Age (`age`).

### Step 2: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Or with explicit port and host:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Open your browser at:
`http://localhost:8501`

---

## 🩺 Dataset Schema (UCI Cleveland Clinic)

| Feature | Description | Range / Values |
|---|---|---|
| `age` | Patient chronological age | 29 – 77 years |
| `sex` | Biological sex | `1 = Male`, `0 = Female` |
| `cp` | Chest pain classification | `0 = Typical Angina`, `1 = Atypical Angina`, `2 = Non-Anginal`, `3 = Asymptomatic` |
| `trestbps` | Resting blood pressure | 94 – 200 mm Hg |
| `chol` | Serum cholesterol | 126 – 564 mg/dL |
| `fbs` | Fasting blood sugar > 120 mg/dL | `1 = True`, `0 = False` |
| `restecg` | Resting electrocardiogram | `0 = Normal`, `1 = ST-T Wave Abnormality`, `2 = Left Ventricular Hypertrophy` |
| `thalach` | Maximum achieved heart rate | 71 – 202 bpm |
| `exang` | Exercise-induced angina | `1 = Yes`, `0 = No` |
| `oldpeak` | ST depression induced by exercise vs rest | 0.0 – 6.2 mm |
| `target` | Diagnosis classification | `0 = Healthy`, `1 = Heart Disease` |

---

## 🔍 "Why Did AI Predict This?" Explainability Mechanism

The system traverses the decision tree in real time for any patient profile:
1. `clf.decision_path(sample)` retrieves the precise path node indices.
2. For each node, the patient's metric is compared against the learned split threshold.
3. The interface renders:
   - **Step Tag & Node ID** (e.g. `Step 1 • Node #0`)
   - **Rule Evaluation** (e.g. `Patient cp = 3 > Threshold 2.50`)
   - **Branch Direction** (`RIGHT: Branch Condition Exceeded`)
   - **Clinical Context** explaining the cardiovascular physiological implications.
   - **Terminal Leaf Distribution** displaying historical training cohort sample counts and probabilities at that exact leaf.

