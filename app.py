import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --------------------------
st.title("Model Accuracy Comparison (Auto Computed)")
st.write("Accuracy calculated from dataset predictions")

# --------------------------
# Load Data
# --------------------------
df = pd.read_csv("results.csv")

y_true = df["y_true"]

# --------------------------
# Compute Accuracies
# --------------------------
accuracy_data = {
    "Naive Bayes": accuracy_score(y_true, df["nb"]),
    "SVM": accuracy_score(y_true, df["svm"]),
    "Decision Tree": accuracy_score(y_true, df["dt"]),
    "CNN": accuracy_score(y_true, df["cnn"]),
    "Vision Transformer": accuracy_score(y_true, df["vit"]),
}

acc_df = pd.DataFrame(list(accuracy_data.items()), columns=["Algorithm", "Accuracy"])

# --------------------------
# Display Table
# --------------------------
st.subheader("Accuracy Table")
st.dataframe(acc_df)

# --------------------------
# Best Model
# --------------------------
best_model = acc_df.loc[acc_df["Accuracy"].idxmax()]
st.success(f"🏆 Best Model: {best_model['Algorithm']} with Accuracy {best_model['Accuracy']:.2f}")

# --------------------------
# Bar Chart
# --------------------------
fig, ax = plt.subplots()
ax.bar(acc_df["Algorithm"], acc_df["Accuracy"])
ax.set_xlabel("Models")
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")

st.pyplot(fig)
