import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Title
# --------------------------
st.title("Model Accuracy Comparison")
st.write("Accuracy comparison of Machine Learning and Deep Learning models for Bacterial Colony Classification")

# --------------------------
# Accuracy Values (PUT YOUR REAL VALUES HERE)
# --------------------------
data = {
    "Algorithm": ["Naive Bayes", "SVM", "Decision Tree", "CNN"],
    "Accuracy": [0.35, 0.70, 0.60, 0.70]   # 🔴 change if needed
}

df = pd.DataFrame(data)

# --------------------------
# Show Table
# --------------------------
st.subheader("Accuracy Table")
st.dataframe(df)

# --------------------------
# Find Best Model
# --------------------------
best_model = df.loc[df["Accuracy"].idxmax()]

st.success(f"Best Model: {best_model['Algorithm']} with Accuracy {best_model['Accuracy']}")

# --------------------------
# Bar Chart
# --------------------------
st.subheader("Accuracy Comparison Chart")

fig, ax = plt.subplots()
ax.bar(df["Algorithm"], df["Accuracy"])
ax.set_xlabel("Models")
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")

st.pyplot(fig)
