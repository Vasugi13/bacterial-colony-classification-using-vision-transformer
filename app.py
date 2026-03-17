import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Title
# --------------------------
st.title("Model Accuracy Comparison")
st.write("Comparison of Machine Learning and Deep Learning models for Bacterial Colony Classification")

# --------------------------
# Accuracy Values (UPDATE IF NEEDED)
# --------------------------
data = {
    "Algorithm": [
        "Naive Bayes",
        "SVM",
        "Decision Tree",
        "CNN",
        "Vision Transformer"
    ],
    "Accuracy": [
        0.35,
        0.70,
        0.55,
        0.50,
        0.75   # 👈 Your ViT accuracy
    ]
}

df = pd.DataFrame(data)

# --------------------------
# Show Table
# --------------------------
st.subheader("Accuracy Table")
st.dataframe(df)

# --------------------------
# Best Model
# --------------------------
best_model = df.loc[df["Accuracy"].idxmax()]

st.success(f"🏆 Best Model: {best_model['Algorithm']} with Accuracy {best_model['Accuracy']}")

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
