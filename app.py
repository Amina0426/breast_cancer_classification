import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model_data = joblib.load("model/breast_cancer_model.pkl")

if isinstance(model_data, dict):
    model = model_data["model"]
    feature_names = model_data["features"]
else:
    model = model_data
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean",
        "area_mean", "smoothness_mean", "compactness_mean",
        "concavity_mean", "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se",
        "fractal_dimension_se", "radius_worst",
        "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]

# --------------------------------------------------
# Prepare example profiles
# --------------------------------------------------

dataset = load_breast_cancer()

# sklearn dataset labels:
# 0 = malignant, 1 = benign
# Select a sample from each class.
benign_indices = np.flatnonzero(dataset.target == 1)
malignant_indices = np.flatnonzero(dataset.target == 0)

benign_sample = dataset.data[benign_indices[len(benign_indices) // 2]]
malignant_sample = dataset.data[
    malignant_indices[len(malignant_indices) // 2]
]

# The sklearn dataset uses the same 30 features in the
# same order as the standard Wisconsin diagnostic dataset.
examples = {
    "Benign — dataset example": dict(
        zip(feature_names, benign_sample.tolist())
    ),
    "Malignant — dataset example": dict(
        zip(feature_names, malignant_sample.tolist())
    )
}

# Initialize the app with a prefilled benign example.
if "example_profile" not in st.session_state:
    st.session_state["example_profile"] = (
        "Benign — dataset example"
    )

if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = 1001

for feature in feature_names:
    key = f"input_{feature}"
    if key not in st.session_state:
        st.session_state[key] = float(
            examples["Benign — dataset example"][feature]
        )

# Update all measurement inputs when the profile changes.
def load_example():
    selected = st.session_state["example_profile"]

    if selected in examples:
        for feature in feature_names:
            st.session_state[f"input_{feature}"] = float(
                examples[selected][feature]
            )

# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("🩺 Breast Cancer Prediction")

st.write(
    "Explore a machine-learning model that classifies "
    "tumor measurements as benign or malignant."
)

st.info(
    "Educational demonstration only. Predictions are not "
    "medical diagnoses and must not guide clinical decisions."
)

# --------------------------------------------------
# Example profile selector
# --------------------------------------------------

st.subheader("Try an Example")

st.caption(
    "Choose an example to automatically fill all 30 measurements. "
    "You can edit any value before predicting."
)

st.selectbox(
    "Example profile",
    options=[
        "Benign — dataset example",
        "Malignant — dataset example",
        "Custom — edit measurements"
    ],
    key="example_profile",
    on_change=load_example
)

if st.session_state["example_profile"] != "Custom — edit measurements":
    st.caption(
        "The displayed measurements are from an example in the "
        "Wisconsin Diagnostic Breast Cancer dataset, not a new patient."
    )

# --------------------------------------------------
# Patient information
# --------------------------------------------------

st.subheader("Patient Information")

st.number_input(
    "Demo Patient ID",
    min_value=0,
    step=1,
    key="patient_id"
)

# --------------------------------------------------
# Feature inputs
# --------------------------------------------------

st.subheader("Tumor Measurements")

inputs = {}
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        inputs[feature] = st.number_input(
            feature.replace("_", " ").title(),
            key=f"input_{feature}",
            format="%.6f"
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", type="primary", use_container_width=True):

    input_data = pd.DataFrame(
        [[inputs[feature] for feature in feature_names]],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Preserve the label convention used by the original app:
        # 0 = benign, 1 = malignant.
        classes = list(model.classes_)

        if 0 not in classes or 1 not in classes:
            st.error(
                f"Unexpected model classes: {classes}. "
                "Check how the target labels were encoded during training."
            )
            st.stop()

        benign_probability = probabilities[classes.index(0)]
        malignant_probability = probabilities[classes.index(1)]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🔴 Prediction: Malignant")
            st.write(
                f"Malignant class probability: "
                f"**{malignant_probability:.2%}**"
            )
        else:
            st.success("🟢 Prediction: Benign")
            st.write(
                f"Benign class probability: "
                f"**{benign_probability:.2%}**"
            )

        st.caption(
            "These probabilities reflect the model's output, "
            "not a clinically validated estimate of an individual's risk."
        )

        # Probability breakdown
        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Diagnosis": ["Benign", "Malignant"],
            "Probability": [
                benign_probability,
                malignant_probability
            ]
        })

        st.bar_chart(
            probability_df.set_index("Diagnosis")
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
