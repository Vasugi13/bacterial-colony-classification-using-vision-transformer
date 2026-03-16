# =========================================
# Bacterial Colony Classification - Streamlit (Optimized)
# =========================================

import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("Bacterial Colony Classification")

# -----------------------------
# 1. Download Dataset from Drive (if needed)
# -----------------------------
dataset_url = "https://drive.google.com/uc?id=1CrYCeMSiuol01NKgDZ5KBVJgFdQKppvi"
dataset_zip = "dataset.zip"

if not os.path.exists("dataset"):
    with st.spinner("Downloading dataset..."):
        gdown.download(dataset_url, dataset_zip, quiet=False)
        os.system("unzip -q dataset.zip -d dataset")

dataset_path = "dataset"

# -----------------------------
# 2. Load Dataset (for labels only)
# -----------------------------
selected_classes = [
    "Escherichia.coli",
    "Staphylococcus.aureus",
    "Candida.albicans",
    "Lactobacillus.plantarum",
    "Enterococcus.faecalis"
]

labels = []
for class_name in selected_classes:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.exists(class_dir):
        labels.append(class_name)

# Encode labels
le = LabelEncoder()
le.fit(labels)
num_classes = len(labels)

img_size = 128

# -----------------------------
# 3. Load Pre-trained Models
# -----------------------------
st.info("Loading pre-trained models...")

cnn = load_model("cnn_model.h5")  # Place cnn_model.h5 in repo or Drive
vit = load_model("vit_model.h5")  # Place vit_model.h5 in repo or Drive

st.success("Models loaded successfully!")

# -----------------------------
# 4. Upload Image for Prediction
# -----------------------------
st.subheader("Upload Colony Image for Prediction")
uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_input = img_resized.reshape(1, img_size, img_size, 3)

    # Use CNN for prediction
    pred_cnn = np.argmax(cnn.predict(img_input), axis=1)
    label_cnn = le.inverse_transform(pred_cnn)

    # Use ViT for prediction
    pred_vit = np.argmax(vit.predict(img_input), axis=1)
    label_vit = le.inverse_transform(pred_vit)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Colony Type (CNN): {label_cnn[0]}")
    st.success(f"Predicted Colony Type (ViT): {label_vit[0]}")

# -----------------------------
# 5. Optional: Display Accuracy Table (static)
# -----------------------------
results = pd.DataFrame({
    "Algorithm": ["CNN","Vision Transformer"],
    "Accuracy": [0.75, 0.70]  # Replace with your precomputed accuracies
})

st.subheader("Model Accuracy Comparison")
st.dataframe(results)

fig, ax = plt.subplots()
ax.bar(results["Algorithm"], results["Accuracy"])
ax.set_ylabel("Accuracy")
ax.set_ylim(0,1)
ax.set_title("Accuracy Comparison")
st.pyplot(fig)
