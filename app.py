import streamlit as st
import pandas as pd
import joblib


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

# If you saved only the pipeline:
if isinstance(model_data, dict):
    model = model_data["model"]
    feature_names = model_data["features"]
else:
    model = model_data

    feature_names = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
    ]


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩺 Breast Cancer Prediction")
st.write(
    "Enter the patient's measurements below to predict whether "
    "the tumor is benign or malignant."
)

st.info(
    "This application is a machine-learning demonstration and "
    "is not a medical diagnostic tool."
)


# --------------------------------------------------
# Patient ID
# --------------------------------------------------

st.subheader("Patient Information")

patient_id = st.number_input(
    "Patient ID",
    min_value=0,
    value=0,
    step=1
)


# --------------------------------------------------
# Feature inputs
# --------------------------------------------------

st.subheader("Tumor Measurements")

inputs = {}

# Create 3 columns for a cleaner layout
cols = st.columns(3)

for i, feature in enumerate(feature_names):

    with cols[i % 3]:
        inputs[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", type="primary"):

    # IMPORTANT:
    # ID is NOT passed to the model.
    # The model was trained only on the 30 features.

    input_data = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]

    benign_probability = probabilities[0]
    malignant_probability = probabilities[1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🔴 Prediction: Malignant")

        st.write(
            f"Malignant probability: "
            f"**{malignant_probability:.2%}**"
        )

    else:

        st.success("🟢 Prediction: Benign")

        st.write(
            f"Benign probability: "
            f"**{benign_probability:.2%}**"
        )

    # --------------------------------------------------
    # Probability breakdown
    # --------------------------------------------------

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