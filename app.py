import streamlit as st
import cv2
import numpy as np
import joblib
import os
import gdown
from tensorflow.keras.models import load_model

# --------------------------
# App Title
# --------------------------
st.title("Bacterial Colony Classification")
st.write("Upload a bacterial colony image to predict the colony type.")

img_size = 128

# --------------------------
# Google Drive Download
# --------------------------
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# --------------------------
# Download ONLY REQUIRED MODELS
# --------------------------
download_model("11RSSrvV4cV9HGNRVMU6bQdHThYOImvw7", "svm.pkl")
download_model("1Xk1VDm3HBW5f9fvgc5JhCqUrs5OXAAt1", "cnn_model.h5")

# ❌ DO NOT LOAD VIT (too large)
# download_model("1vDSCeF_O3WpS98Tidpcm71t-1voiozX3", "vit_model.h5")

# --------------------------
# Load Models Safely
# --------------------------
@st.cache_resource
def load_models():
    svm = nb = dt = cnn = le = None

    # GitHub models
    try:
        nb = joblib.load("naive_bayes.pkl")
    except:
        st.warning("⚠️ Naive Bayes failed")

    try:
        dt = joblib.load("decision_tree.pkl")
    except:
        st.warning("⚠️ Decision Tree failed")

    try:
        le = joblib.load("label_encoder.pkl")
    except:
        st.error("❌ Label Encoder missing")

    # Drive models
    try:
        svm = joblib.load("svm.pkl")
    except:
        st.warning("⚠️ SVM failed")

    try:
        cnn = load_model("cnn_model.h5")
    except:
        st.error("❌ CNN failed")

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
        if nb:
            st.write("Naive Bayes:", le.inverse_transform(nb.predict(image_flat))[0])
        else:
            st.write("Naive Bayes: ❌")

        # SVM
        if svm:
            st.write("SVM:", le.inverse_transform(svm.predict(image_flat))[0])
        else:
            st.write("SVM: ❌")

        # DT
        if dt:
            st.write("Decision Tree:", le.inverse_transform(dt.predict(image_flat))[0])
        else:
            st.write("Decision Tree: ❌")

        # CNN
        if cnn:
            pred = np.argmax(cnn.predict(image_dl), axis=1)
            st.write("CNN:", le.inverse_transform(pred)[0])
        else:
            st.write("CNN: ❌")

    else:
        st.error("Label encoder missing!")
