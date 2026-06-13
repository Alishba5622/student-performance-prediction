import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Predictor", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown("### Machine Learning Dashboard (Naïve Bayes + Neural Network)")

# =========================
# DATASET
# =========================
X = np.array([
    [2,50,3],[3,60,4],[1,40,2],[2,55,3],[3,62,4],
    [4,70,5],[5,80,7],[6,90,8],[5,85,7],[6,88,9],
    [4,72,6],[7,95,10],[5,78,7],[6,82,8],[4,75,6]
])

y = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

df = pd.DataFrame(X, columns=["Study Hours","Attendance","Assignments"])
df["Result"] = y

# =========================
# MODELS
# =========================
nb = GaussianNB().fit(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

nn = MLPClassifier(hidden_layer_sizes=(6,), max_iter=800, random_state=42)
nn.fit(X_scaled, y)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("📌 Input Student Data")

study = st.sidebar.slider("Study Hours", 1, 10, 4)
attendance = st.sidebar.slider("Attendance %", 0, 100, 75)
assignments = st.sidebar.slider("Assignments", 0, 10, 6)

input_data = np.array([[study, attendance, assignments]])

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "📁 Dataset"])

# =========================
# TAB 1 - DASHBOARD
# =========================
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Pass Students", sum(y))
    col3.metric("Fail Students", len(y)-sum(y))

    st.dataframe(df)

    st.subheader("Feature Distribution")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Study Hours", y="Attendance", hue="Result", ax=ax)
    st.pyplot(fig)

# =========================
# TAB 2 - PREDICTION
# =========================
with tab2:
    st.subheader("🔮 Live Prediction System")

    nb_pred = nb.predict(input_data)[0]
    nn_pred = nn.predict(scaler.transform(input_data))[0]

    final = 1 if (nb_pred + nn_pred) >= 1 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Naïve Bayes", "PASS" if nb_pred else "FAIL")
    col2.metric("Neural Network", "PASS" if nn_pred else "FAIL")
    col3.metric("Final Decision", "PASS" if final else "FAIL")

    confidence = (nb_pred + nn_pred) / 2 * 100
    st.progress(int(confidence))

    st.success(f"Model Confidence: {confidence:.0f}%")

# =========================
# TAB 3 - CONFUSION MATRIX
# =========================
with tab3:
    st.subheader("Model Evaluation")

    nb_cm = confusion_matrix(y, nb.predict(X))

    fig2, ax2 = plt.subplots()
    sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title("Naïve Bayes Confusion Matrix")

    st.pyplot(fig2)

    st.info("Neural Network learns better patterns but needs more data.")