import streamlit as st
import cv2
import numpy as np
import joblib
import gdown
import os
from tensorflow.keras.models import load_model

# --------------------------
st.title("Bacterial Colony Classification")
st.write("Upload a bacterial colony image to predict the colony type.")

img_size = 128

# -----------------------------
# Download Models (ONLY DL MODELS)
# -----------------------------
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# Download only CNN (SAFE)
download_model("1Xk1VDm3HBW5f9fvgc5JhCqUrs5OXAAt1", "cnn_model.h5")

# ⚠️ ViT is very large → skip or keep optional
# download_model("1vDSCeF_O3WpS98Tidpcm71t-1voiozX3", "vit_model.h5")


# -----------------------------
# Load Models (SAFE LOADING)
# -----------------------------
@st.cache_resource
def load_all_models():

    svm = nb = dt = le = None
    cnn = vit = None

    # ---------------- ML MODELS ----------------
    try:
        svm = joblib.load("svm.pkl")
    except:
        st.warning("⚠️ SVM model failed to load")

    try:
        nb = joblib.load("naive_bayes.pkl")
    except:
        st.warning("⚠️ Naive Bayes model failed to load")

    try:
        dt = joblib.load("decision_tree.pkl")
    except:
        st.warning("⚠️ Decision Tree model failed to load")

    try:
        le = joblib.load("label_encoder.pkl")
    except:
        st.error("❌ Label Encoder missing")

    # ---------------- DL MODELS ----------------
    try:
        cnn = load_model("cnn_model.h5")
    except:
        st.error("❌ CNN model failed to load")

    # Optional ViT (skip if not needed)
    if os.path.exists("vit_model.h5"):
        try:
            vit = load_model("vit_model.h5")
        except:
            st.warning("⚠️ ViT model failed / too large → skipped")

    return svm, nb, dt, cnn, vit, le


svm, nb, dt, cnn, vit, le = load_all_models()

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Colony Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image_resized = cv2.resize(image, (img_size, img_size))

    image_flat = image_resized.reshape(1, -1)
    image_dl = image_resized.reshape(1, img_size, img_size, 3)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Prediction Results")

    # -----------------------------
    # ML Predictions (SAFE)
    # -----------------------------
    if le is not None:

        if nb is not None:
            nb_pred = le.inverse_transform(nb.predict(image_flat))[0]
            st.write("Naive Bayes:", nb_pred)
        else:
            st.write("Naive Bayes: ❌ Not Available")

        if svm is not None:
            svm_pred = le.inverse_transform(svm.predict(image_flat))[0]
            st.write("SVM:", svm_pred)
        else:
            st.write("SVM: ❌ Not Available")

        if dt is not None:
            dt_pred = le.inverse_transform(dt.predict(image_flat))[0]
            st.write("Decision Tree:", dt_pred)
        else:
            st.write("Decision Tree: ❌ Not Available")

        # -----------------------------
        # CNN Prediction
        # -----------------------------
        if cnn is not None:
            cnn_pred = le.inverse_transform(
                np.argmax(cnn.predict(image_dl), axis=1)
            )[0]
            st.write("CNN:", cnn_pred)
        else:
            st.write("CNN: ❌ Not Available")

        # -----------------------------
        # ViT Prediction (OPTIONAL)
        # -----------------------------
        if vit is not None:
            vit_pred = le.inverse_transform(
                np.argmax(vit.predict(image_dl), axis=1)
            )[0]
            st.write("Vision Transformer:", vit_pred)
        else:
            st.write("Vision Transformer: ⚠️ Skipped")

    else:
        st.error("❌ Label Encoder not loaded → Cannot show predictions")
