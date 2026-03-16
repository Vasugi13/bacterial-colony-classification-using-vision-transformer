import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gdown

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.title("Bacterial Colony Classification")

# -----------------------------
# Download Dataset from Drive
# -----------------------------
url = "https://drive.google.com/uc?id=1CrYCeMSiuol01NKgDZ5KBVJgFdQKppvi"
output = "dataset.zip"

if not os.path.exists("dataset"):
    with st.spinner("Downloading Dataset..."):
        gdown.download(url, output, quiet=False)
        os.system("unzip dataset.zip")

dataset_path = "dataset"

# -----------------------------
# Load Dataset
# -----------------------------
images = []
labels = []

img_size = 64

for folder in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(class_path):
        continue

    for img in os.listdir(class_path):

        img_path = os.path.join(class_path, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (img_size, img_size))

        images.append(image)
        labels.append(folder)

X = np.array(images)
y = np.array(labels)

# -----------------------------
# Encode Labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Flatten
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# -----------------------------
# Train Models
# -----------------------------
nb = GaussianNB()
nb.fit(X_train_flat, y_train)
nb_pred = nb.predict(X_test_flat)
nb_acc = accuracy_score(y_test, nb_pred)

svm = SVC()
svm.fit(X_train_flat, y_train)
svm_pred = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test, svm_pred)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train_flat, y_train)
dt_pred = dt.predict(X_test_flat)
dt_acc = accuracy_score(y_test, dt_pred)

# -----------------------------
# Show Results
# -----------------------------
results = pd.DataFrame({
    "Algorithm": ["Naive Bayes", "SVM", "Decision Tree"],
    "Accuracy": [nb_acc, svm_acc, dt_acc]
})

st.subheader("Model Accuracy Comparison")
st.dataframe(results)

# -----------------------------
# Accuracy Graph
# -----------------------------
fig, ax = plt.subplots()
ax.bar(results["Algorithm"], results["Accuracy"])
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")

st.pyplot(fig)

# -----------------------------
# Upload Image for Prediction
# -----------------------------
st.subheader("Upload Colony Image for Prediction")

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    image = cv2.resize(image, (img_size, img_size))
    image_flat = image.reshape(1, -1)

    prediction = svm.predict(image_flat)
    label = le.inverse_transform(prediction)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Colony Type: {label[0]}")
