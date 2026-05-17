import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Loan Predictor", layout="wide")

st.title("🏦 Smart Loan Prediction System")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

data = load_data()

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# ---------------- DATA CLEANING ----------------
st.subheader("🧹 Data Cleaning")

# Drop Loan_ID (string column)
if "Loan_ID" in data.columns:
    data = data.drop("Loan_ID", axis=1)

# Convert categorical to numeric
data.replace({
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
    'Loan_Status': {'Y': 1, 'N': 0}
}, inplace=True)

# Fix Dependents column
if "Dependents" in data.columns:
    data['Dependents'] = data['Dependents'].replace('3+', 3)
    data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce')

# Drop missing values
data = data.dropna()

st.write("Cleaned Data Shape:", data.shape)

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    data['Loan_Status'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title("Loan Status Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    data['ApplicantIncome'].plot(kind='hist', bins=30, ax=ax2)
    ax2.set_title("Applicant Income Distribution")
    st.pyplot(fig2)

# ---------------- MODEL TRAINING ----------------
st.subheader("🤖 Model Training")

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- PERFORMANCE ----------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# ---------------- PREDICTION ----------------
st.subheader("🔮 Predict Loan Approval")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
    Married = st.selectbox("Married (0=No, 1=Yes)", [0, 1])
    Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    Education = st.selectbox("Education (0=Not Graduate, 1=Graduate)", [0, 1])
    Self_Employed = st.selectbox("Self Employed (0=No, 1=Yes)", [0, 1])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0)

with col2:
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0.0)
    Credit_History = st.selectbox("Credit History", [0, 1])
    Property_Area = st.selectbox("Property Area (0=Rural,1=Semiurban,2=Urban)", [0, 1, 2])

# Prediction button
if st.button("Predict Loan Status"):
    input_data = np.array([[Gender, Married, Dependents, Education,
                            Self_Employed, ApplicantIncome, CoapplicantIncome,
                            LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")