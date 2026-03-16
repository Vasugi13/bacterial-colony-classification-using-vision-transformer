import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import zipfile

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

st.title("Bacterial Colony Classification")

# -----------------------------
# 1. Download Dataset from Google Drive
# -----------------------------
dataset_url = "https://drive.google.com/uc?id=1CrYCeMSiuol01NKgDZ5KBVJgFdQKppvi"
dataset_zip = "dataset.zip"
dataset_folder = "dataset"

if not os.path.exists(dataset_folder):
    with st.spinner("Downloading dataset..."):
        gdown.download(dataset_url, dataset_zip, quiet=False)
    with st.spinner("Extracting dataset..."):
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)

st.success("Dataset ready!")

# -----------------------------
# 2. Select Classes (Optional)
# -----------------------------
selected_classes = [
    "Escherichia.coli",
    "Staphylococcus.aureus",
    "Candida.albicans",
    "Lactobacillus.plantarum",
    "Enterococcus.faecalis"
]

# -----------------------------
# 3. Load Dataset
# -----------------------------
images = []
labels = []
img_size = 128

for class_name in selected_classes:
    class_path = os.path.join(dataset_folder, class_name)
    if not os.path.exists(class_path):
        st.warning(f"Folder not found: {class_path}")
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(class_name)

if len(images) == 0:
    st.error("No images found in dataset!")
    st.stop()

X = np.array(images)
y = np.array(labels)
st.write(f"Loaded {len(X)} images.")

# -----------------------------
# 4. Encode Labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Flatten Data for ML Models
# -----------------------------
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# -----------------------------
# 7. Classical ML Models
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
# 8. CNN Model
# -----------------------------
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
with st.spinner("Training CNN..."):
    cnn.fit(X_train, y_train_cat, epochs=10, batch_size=16, validation_data=(X_test, y_test_cat))
cnn_pred = np.argmax(cnn.predict(X_test), axis=1)
cnn_acc = accuracy_score(y_test, cnn_pred)

# -----------------------------
# 9. Simple Vision Transformer
# -----------------------------
inputs = layers.Input(shape=(img_size,img_size,3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(64,3,activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(128,activation="relu")(x)
outputs = layers.Dense(num_classes,activation="softmax")(x)
vit_model = models.Model(inputs, outputs)
vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
with st.spinner("Training Vision Transformer..."):
    vit_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
vit_pred = np.argmax(vit_model.predict(X_test), axis=1)
vit_acc = accuracy_score(y_test, vit_pred)

# -----------------------------
# 10. Accuracy Table & Graph
# -----------------------------
results = pd.DataFrame({
    "Algorithm": ["Bayesian","SVM","Decision Tree","CNN","Vision Transformer"],
    "Accuracy": [nb_acc, svm_acc, dt_acc, cnn_acc, vit_acc]
})

st.subheader("Model Accuracy Comparison")
st.dataframe(results)

fig, ax = plt.subplots()
ax.bar(results["Algorithm"], results["Accuracy"])
ax.set_ylim(0,1)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison")
st.pyplot(fig)

# -----------------------------
# 11. Upload Image for Prediction
# -----------------------------
st.subheader("Upload Colony Image for Prediction")
model_choice = st.selectbox("Select Model", ["Bayesian","SVM","Decision Tree","CNN","Vision Transformer"])
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (img_size,img_size))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_choice in ["Bayesian","SVM","Decision Tree"]:
        image_flat = image.reshape(1, -1)
        if model_choice == "Bayesian":
            pred = nb.predict(image_flat)
        elif model_choice == "SVM":
            pred = svm.predict(image_flat)
        else:
            pred = dt.predict(image_flat)
    else:
        image_input = image.reshape(1,img_size,img_size,3)
        if model_choice == "CNN":
            pred = np.argmax(cnn.predict(image_input), axis=1)
        else:
            pred = np.argmax(vit_model.predict(image_input), axis=1)
    label = le.inverse_transform(pred)
    st.success(f"Predicted Colony Type ({model_choice}): {label[0]}")
