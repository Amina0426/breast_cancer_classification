# 🩺 Breast Cancer Classification

A machine learning classification project that predicts whether a breast tumor is **Benign** or **Malignant** using the Wisconsin Diagnostic Breast Cancer dataset.

The project covers the complete machine learning workflow from exploratory data analysis and model evaluation to pipeline creation, model serialization, and deployment with Streamlit.

## 🚀 Live Demo

🔗 **[Try the Breast Cancer Prediction App](https://breastcancerclassification-amina.streamlit.app/)**

---

## 📌 Project Overview

The model uses 30 numerical features describing characteristics of cell nuclei obtained from digitized fine needle aspirate (FNA) images.

### Target

- `0` → Benign
- `1` → Malignant

The `id` column was removed because it is an identifier and does not provide predictive information. The `Unnamed: 32` column was also removed because it contains no useful information.

Detailed exploratory analysis and experimentation are documented in the Jupyter notebook.

---

## 📊 Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**

- 569 observations
- 30 numerical features
- Binary classification problem
- No missing values
- No duplicate observations

The 30 features describe measurements such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each measurement is provided as `mean`, `se`, and `worst` values.

---

## 🤖 Model Comparison

Several classification algorithms were evaluated using cross-validation, hyperparameter tuning, and a held-out test set.

| Model                   |   Accuracy |   Precision |     Recall |   F1 Score |
| ----------------------- | ---------: | ----------: | ---------: | ---------: |
| **Logistic Regression** | **96.49%** |      95.24% | **95.24%** |     95.24% |
| K-Nearest Neighbors     |     94.74% |     100.00% |     85.71% |     92.31% |
| SVC                     | **97.37%** | **100.00%** |     92.86% | **96.30%** |
| Random Forest           |     95.61% |      95.12% |     92.86% |     93.98% |

Although SVC achieved slightly higher accuracy and F1 score, **Logistic Regression was selected as the final model because it achieved higher recall for malignant cases**.

---

## 🎯 Final Model

The final model consists of:

**StandardScaler → Logistic Regression**

### Hyperparameters

`LogisticRegression(C=1, max_iter=100)`

### Final Test Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **96.49%** |
| Precision | **95.24%** |
| Recall    | **95.24%** |
| F1 Score  | **95.24%** |
| ROC-AUC   |  **0.987** |

### Confusion Matrix

|              | Predicted B | Predicted M |
| ------------ | ----------: | ----------: |
| **Actual B** |          70 |           2 |
| **Actual M** |           2 |          40 |

The model correctly identified **40 out of 42 malignant cases** in the held-out test set.

---

## 🔬 Feature Selection & PCA

Dimensionality reduction was also explored to determine whether fewer features could achieve comparable or better performance.

| Approach           | Features / Components |   Accuracy |  Precision |     Recall |         F1 |
| ------------------ | --------------------: | ---------: | ---------: | ---------: | ---------: |
| **Original Model** |                    30 | **96.49%** | **95.24%** | **95.24%** | **95.24%** |
| SelectKBest        |                    20 |     95.61% |     93.02% |     95.24% |     94.12% |
| PCA                |               Reduced |     95.61% |     93.02% |     95.24% |     94.12% |

The original 30-feature model was retained because dimensionality reduction did not improve test-set performance.

---

## 📈 ROC-AUC

The final Logistic Regression model achieved a **ROC-AUC of 0.987**, indicating strong discriminatory performance between benign and malignant cases.

---

## 💾 Model Serialization

The final preprocessing and classification steps were combined into a Scikit-learn Pipeline and serialized using `joblib`.

The serialized model is stored at:

`model/breast_cancer_model.pkl`

The pipeline contains both:

- `StandardScaler`
- `LogisticRegression`

This ensures that the same preprocessing is automatically applied when making predictions.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can enter the 30 tumor measurements and receive:

- Benign or Malignant prediction
- Prediction probability
- Probability visualization

The patient `id` can be entered for identification, but it is not passed to the machine learning model.

### Application Flow

**User Input → 30 Tumor Measurements → Serialized Pipeline → StandardScaler → Logistic Regression → Prediction + Probability**

---

## 🛠️ Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Logistic Regression
- SVC
- KNN
- Random Forest
- Gaussian Naive Bayes
- Decision Tree

### Deployment

- Streamlit
- Joblib

### Development

- Jupyter Notebook
- VS Code
- Python Virtual Environment

---

## ▶️ Run Locally

### 1. Clone the repository

    git clone https://github.com/Amina0426/breast_cancer_classification.git
    cd breast-cancer

### 2. Create a virtual environment

    python -m venv venv

### 3. Activate the virtual environment

Windows:

    venv\Scripts\activate

### 4. Install dependencies

    pip install -r requirements.txt

### 5. Run the Streamlit application

    streamlit run app.py

The application will open in your browser at the local Streamlit URL.

---

## 📓 Documentation

The Jupyter notebook contains the detailed analysis and experimentation performed throughout the project, including:

- Exploratory Data Analysis
- Data preprocessing
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- ROC-AUC analysis
- Feature interpretation
- Feature selection
- PCA experimentation
- Pipeline creation
- Model serialization

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**.

It is not a medical diagnostic system and should not be used for clinical decision-making. Predictions from this application should not replace evaluation by qualified healthcare professionals.

---

## 👩‍💻 Author

**Amina**
