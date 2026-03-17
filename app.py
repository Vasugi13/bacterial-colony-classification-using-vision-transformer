import streamlit as st
import cv2
import numpy as np
import joblib
import os
import gdown
from tensorflow.keras.models import load_model

# --------------------------
# Title
# --------------------------
st.title("Bacterial Colony Classification")
st.write("Upload a bacterial colony image to predict the colony type.")

img_size = 128

# --------------------------
# SAFE DOWNLOAD FUNCTION
# --------------------------
def download_model(file_id, output):
    if not os.path.exists(output):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False, fuzzy=True)
        except:
            st.error(f"❌ Failed to download {output}")

# --------------------------
# DOWNLOAD MODELS FROM DRIVE
# --------------------------
# SVM
download_model("11RSSrvV4cV9HGNRVMU6bQdHThYOImvw7", "svm.pkl")

# CNN
download_model("1Xk1VDm3HBW5f9fvgc5JhCqUrs5OXAAt1", "cnn_model.h5")

# ❌ DO NOT USE VIT (too large)

# --------------------------
# LOAD MODELS
# --------------------------
@st.cache_resource
def load_models():
    svm = nb = dt = cnn = le = None

    try:
        svm = joblib.load("svm.pkl")
    except:
        st.error("❌ SVM not loaded")

    try:
        nb = joblib.load("naive_bayes.pkl")
    except:
        st.error("❌ Naive Bayes not loaded")

    try:
        dt = joblib.load("decision_tree.pkl")
    except:
        st.error("❌ Decision Tree not loaded")

    try:
        le = joblib.load("label_encoder.pkl")
    except:
        st.error("❌ Label Encoder missing")

    try:
        cnn = load_model("cnn_model.h5")
    except:
        st.error("❌ CNN not loaded")

    return svm, nb, dt, cnn, le


svm, nb, dt, cnn, le = load_models()

# --------------------------
# Upload Image
# --------------------------
uploaded_file = st.file_uploader("Upload Colony Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image_resized = cv2.resize(image, (img_size, img_size))

    image_flat = image_resized.reshape(1, -1)
    image_dl = image_resized.reshape(1, img_size, img_size, 3)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.subheader("Prediction Results")

    if le:

        # NB
        try:
            st.write("Naive Bayes:", le.inverse_transform(nb.predict(image_flat))[0])
        except:
            st.write("Naive Bayes: ❌ Error")

        # SVM
        try:
            st.write("SVM:", le.inverse_transform(svm.predict(image_flat))[0])
        except:
            st.write("SVM: ❌ Error")

        # DT
        try:
            st.write("Decision Tree:", le.inverse_transform(dt.predict(image_flat))[0])
        except:
            st.write("Decision Tree: ❌ Error")

        # CNN
        try:
            pred = np.argmax(cnn.predict(image_dl), axis=1)
            st.write("CNN:", le.inverse_transform(pred)[0])
        except:
            st.write("CNN: ❌ Error")

    else:
        st.error("Label encoder missing")
