import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap

# ─── Load model and scaler ────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ─── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ─── Session state for history ───────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ─── FEATURE NAMES ───────────────────────────────────────────────
FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigree', 'Age']

# ─── NORMAL RANGES ───────────────────────────────────────────────
def get_indicator(label, value):
    ranges = {
        "Glucose":        [(0,   99,  "✅ Normal",          "green"),
                           (100, 125, "⚠️ Borderline",      "orange"),
                           (126, 999, "🔴 High",             "red")],
        "BloodPressure":  [(0,   79,  "✅ Normal",          "green"),
                           (80,  89,  "⚠️ Borderline",      "orange"),
                           (90,  999, "🔴 High",             "red")],
        "BMI":            [(0,   24.9,"✅ Normal",          "green"),
                           (25,  29.9,"⚠️ Overweight",      "orange"),
                           (30,  999, "🔴 Obese",            "red")],
        "Insulin":        [(0,   140, "✅ Normal",          "green"),
                           (141, 200, "⚠️ Borderline",      "orange"),
                           (201, 999, "🔴 High",             "red")],
        "Age":            [(0,   35,  "✅ Low Risk Age",    "green"),
                           (36,  50,  "⚠️ Medium Risk Age", "orange"),
                           (51,  999, "🔴 High Risk Age",   "red")],
    }
    if label not in ranges:
        return "", "gray"
    for low, high, status, color in ranges[label]:
        if low <= value <= high:
            return status, color
    return "", "gray"

# ─── HEALTH TIPS ─────────────────────────────────────────────────
def show_health_tips(prediction, glucose, bmi):
    st.markdown("---")
    if prediction == 1:
        st.markdown("### 🩺 Personalized Health Recommendations")
        st.error("You are at **High Risk**. Please consult a doctor.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🥗 Diet**")
            st.markdown("- Cut sugary drinks & refined carbs\n- Eat more vegetables & whole grains\n- Control portion sizes")
            st.markdown("**🏃 Exercise**")
            st.markdown("- At least 30 min walk daily\n- Try swimming or cycling\n- Avoid sitting for long hours")
        with col2:
            st.markdown("**💊 Medical**")
            st.markdown("- Get HbA1c test done\n- Monitor blood sugar weekly\n- Consult an endocrinologist")
            st.markdown("**😴 Lifestyle**")
            st.markdown("- Sleep 7–8 hours daily\n- Reduce stress levels\n- Quit smoking if applicable")
        if glucose > 125:
            st.warning("⚠️ Your glucose level is high. Avoid sugar-heavy foods immediately.")
        if bmi > 30:
            st.warning("⚠️ Your BMI indicates obesity — a major diabetes risk factor.")
    else:
        st.markdown("### 🌿 Keep Up the Good Work!")
        st.success("You are at **Low Risk**. Maintain your healthy lifestyle:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🥗 Diet**\n- Maintain balanced diet\n- Stay hydrated (8 glasses/day)\n- Limit processed food")
        with col2:
            st.markdown("**🏃 Exercise**\n- Keep exercising regularly\n- Annual health checkups\n- Monitor weight")

# ─── FEATURE IMPORTANCE ──────────────────────────────────────────
def show_feature_importance():
    st.markdown("---")
    st.markdown("### 📊 Feature Importance — What Affects Predictions Most?")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_NAMES[i] for i in indices]
    sorted_importances = importances[indices]
    colors = ['#2ECC71' if i == 0 else '#3498DB' if i == 1
              else '#E67E22' if i == 2 else '#BDC3C7'
              for i in range(len(sorted_features))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_features[::-1], sorted_importances[::-1],
            color=colors[::-1], edgecolor='black', height=0.6)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
    for i, val in enumerate(sorted_importances[::-1]):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=10)
    ax.set_xlim(0, max(sorted_importances) + 0.06)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"🥇 **{sorted_features[0]}** is the strongest predictor of diabetes in this model.")

# ─── SHAP EXPLAINABILITY (FULLY FIXED) ───────────────────────────
def show_shap(input_scaled, input_raw):
    st.markdown("---")
    st.markdown("### 🔢 SHAP Explainability — Why This Prediction?")
    st.caption("SHAP values show how much each feature pushed the prediction toward High Risk or Low Risk.")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # Shape is (1, 8, 2) — 1 sample, 8 features, 2 classes
        sv = np.array(shap_values)
        if sv.ndim == 3:
            sv = sv[0, :, 1]   # first sample, all features, class 1 (diabetic)
        elif sv.ndim == 2:
            sv = sv[0]         # first sample
        else:
            sv = sv.flatten()[:len(FEATURE_NAMES)]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#E74C3C' if v > 0 else '#2ECC71' for v in sv]
        y_pos = np.arange(len(FEATURE_NAMES))
        ax.barh(y_pos, sv, color=colors, edgecolor='black', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{FEATURE_NAMES[i]}: {input_raw[i]}" for i in range(len(FEATURE_NAMES))])
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (Red = increases risk, Green = decreases risk)')
        ax.set_title('SHAP Feature Contribution for This Prediction', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("🔴 Red bars push toward **High Risk** | 🟢 Green bars push toward **Low Risk**")

    except Exception as e:
        st.warning(f"SHAP visualization unavailable: {e}")

# ════════════════════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════════════════════
page = st.sidebar.selectbox("📌 Navigate", ["🩺 Predict", "📁 Batch Prediction", "📋 History", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════════
if page == "🩺 Predict":
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Enter the patient's medical details below.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        pregnancies    = st.number_input("Pregnancies", 0, 20, 1)
        glucose        = st.number_input("Glucose Level", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin Level", 0, 900, 80)
        bmi     = st.number_input("BMI", 0.0, 70.0, 32.0)
        dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47)
        age     = st.number_input("Age", 1, 120, 33)

    st.markdown("#### 📋 Your Value Health Indicators")
    ind_cols = st.columns(5)
    checks = [("Glucose", glucose), ("BloodPressure", blood_pressure),
              ("BMI", bmi), ("Insulin", insulin), ("Age", age)]
    for col, (label, val) in zip(ind_cols, checks):
        status, color = get_indicator(label, val)
        col.markdown(f"**{label}**")
        if color == "green":    col.success(status)
        elif color == "orange": col.warning(status)
        else:                   col.error(status)

    st.divider()

    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        input_raw    = [pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]
        input_data   = np.array([input_raw])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]

        st.markdown("### 🔬 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Diabetes — Probability: {probability*100:.1f}%")
        else:
            st.success(f"✅ Low Risk of Diabetes — Probability: {probability*100:.1f}%")
        st.progress(float(probability))
        st.caption(f"{probability*100:.1f}% chance of diabetes based on input values")

        st.session_state.history.append({
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi, "DPF": dpf, "Age": age,
            "Result": "High Risk" if prediction == 1 else "Low Risk",
            "Probability": f"{probability*100:.1f}%"
        })

        report_df = pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi, "DPF": dpf, "Age": age,
            "Result": "High Risk" if prediction == 1 else "Low Risk",
            "Probability": f"{probability*100:.1f}%"
        }])
        st.download_button("📄 Download Report as CSV",
                           report_df.to_csv(index=False),
                           "diabetes_report.csv", "text/csv")

        show_health_tips(prediction, glucose, bmi)
        show_feature_importance()
        show_shap(input_scaled, input_raw)

    st.divider()
    st.caption("Built with scikit-learn + Streamlit | PIMA Indians Diabetes Dataset")

# ════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction — Multiple Patients")
    st.markdown("Upload a CSV file with multiple patients to get predictions for all at once.")
    st.info("Your CSV must have these columns: **Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age**")

    sample = pd.DataFrame({
        'Pregnancies': [1, 3, 6],
        'Glucose': [85, 120, 148],
        'BloodPressure': [66, 70, 72],
        'SkinThickness': [29, 20, 35],
        'Insulin': [0, 80, 0],
        'BMI': [26.6, 31.2, 33.6],
        'DiabetesPedigreeFunction': [0.351, 0.472, 0.627],
        'Age': [31, 28, 50]
    })
    st.download_button("📥 Download Sample CSV Template",
                       sample.to_csv(index=False),
                       "sample_patients.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown("#### Uploaded Data:")
        st.dataframe(df, use_container_width=True)

        if st.button("🔍 Predict for All Patients", use_container_width=True):
            try:
                df_input = df.drop(columns=['Outcome'], errors='ignore')
                expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                df_input = df_input[[c for c in expected_cols if c in df_input.columns]]
                scaled = scaler.transform(df_input.values)
                preds  = model.predict(scaled)
                probs  = model.predict_proba(scaled)[:, 1]
                df_input['Result']      = ['High Risk' if p == 1 else 'Low Risk' for p in preds]
                df_input['Probability'] = [f"{p*100:.1f}%" for p in probs]
                st.markdown("#### 📊 Prediction Results:")
                st.dataframe(df_input, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Patients", len(preds))
                c2.metric("High Risk", int((preds==1).sum()))
                c3.metric("Low Risk",  int((preds==0).sum()))
                st.download_button("📄 Download Results CSV",
                                   df_input.to_csv(index=False),
                                   "batch_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}. Make sure your CSV has the correct columns.")

# ════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ════════════════════════════════════════════════════════════════
elif page == "📋 History":
    st.title("📋 Prediction History")
    if not st.session_state.history:
        st.info("No predictions yet. Go to 🩺 Predict page and make a prediction first!")
    else:
        df_history = pd.DataFrame(st.session_state.history)
        st.markdown(f"**{len(df_history)} prediction(s) made this session:**")
        st.dataframe(df_history, use_container_width=True)
        st.download_button("📄 Download Full History as CSV",
                           df_history.to_csv(index=False),
                           "prediction_history.csv", "text/csv")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("## 🩺 Disease Prediction System\nA machine learning web application that predicts diabetes risk based on clinical data.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset",  "PIMA Indians")
    col2.metric("Records",  "768 patients")
    col3.metric("Features", "8 clinical")
    col4.metric("Model",    "Random Forest")

    st.markdown("---")
    st.markdown("### 🔬 How It Works")
    st.markdown("""
1. **Data Collection** — PIMA Indians Diabetes Dataset (UCI ML Repository)
2. **EDA** — Explored distributions, correlations, and hidden missing values
3. **Preprocessing** — Replaced zero values with median, capped outliers, applied StandardScaler
4. **Model Training** — Compared Logistic Regression, Random Forest, SVM, and KNN
5. **Tuning** — GridSearchCV with 5-fold cross validation (180 combinations tested)
6. **Deployment** — Streamlit Cloud with real-time prediction interface
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.table(pd.DataFrame({
        "Model":    ["Logistic Regression", "Random Forest ⭐", "SVM", "KNN"],
        "Accuracy": ["~78%", "~82%", "~79%", "~75%"],
        "Status":   ["Baseline", "Best Model", "Runner-up", "Baseline"]
    }))

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
| Layer | Technology |
|-------|-----------|
| Language | Python 3 |
| ML Library | scikit-learn |
| Explainability | SHAP |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
    """)
    st.caption("Built by Kyathi Sujitha Dasari | Disease Prediction System | 2025")
