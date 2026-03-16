import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.title("Bacterial Colony Classification")

img_size = 128

# Load models
nb = joblib.load("naive_bayes.pkl")
svm = joblib.load("svm.pkl")
dt = joblib.load("decision_tree.pkl")

cnn = load_model("cnn_model.h5")
vit = load_model("vit_model.h5")

le = joblib.load("label_encoder.pkl")

# Upload Image
uploaded_file = st.file_uploader("Upload Colony Image")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes,1)

    image_resized = cv2.resize(image,(img_size,img_size))

    image_flat = image_resized.reshape(1,-1)

    image_dl = image_resized.reshape(1,img_size,img_size,3)

    # Predictions
    nb_pred = le.inverse_transform(nb.predict(image_flat))[0]
    svm_pred = le.inverse_transform(svm.predict(image_flat))[0]
    dt_pred = le.inverse_transform(dt.predict(image_flat))[0]

    cnn_pred = le.inverse_transform(np.argmax(cnn.predict(image_dl),axis=1))[0]
    vit_pred = le.inverse_transform(np.argmax(vit.predict(image_dl),axis=1))[0]

    st.image(image,caption="Uploaded Image")

    st.subheader("Predictions")

    st.write("Naive Bayes:",nb_pred)
    st.write("SVM:",svm_pred)
    st.write("Decision Tree:",dt_pred)
    st.write("CNN:",cnn_pred)
    st.write("Vision Transformer:",vit_pred)
