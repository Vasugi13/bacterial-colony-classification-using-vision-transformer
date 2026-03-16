import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Title
st.title("Bacterial Colony Classification using Machine Learning and Deep Learning")

st.write("This app compares multiple algorithms for bacterial colony image classification.")

# Dataset path (folder inside GitHub repo)
dataset_path = "bacterial colony dataset"

# Selected bacteria classes
selected_classes = [
"Escherichia.coli",
"Staphylococcus.aureus",
"Candida.albicans",
"Lactobacillus.plantarum",
"Enterococcus.faecalis"
]

# Image size
img_size = 128

# Load dataset
images = []
labels = []

st.write("Loading dataset...")

for class_name in selected_classes:

    class_path = os.path.join(dataset_path,class_name)

    for img in os.listdir(class_path):

        img_path = os.path.join(class_path,img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image,(img_size,img_size))

        images.append(image)
        labels.append(class_name)

X = np.array(images)
y = np.array(labels)

st.write("Dataset Shape:",X.shape)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(np.unique(y_encoded))

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
X,y_encoded,test_size=0.2,random_state=42
)

# Flatten for ML models
X_train_flat = X_train.reshape(X_train.shape[0],-1)
X_test_flat = X_test.reshape(X_test.shape[0],-1)

st.write("Training Machine Learning Models...")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_flat,y_train)

nb_pred = nb.predict(X_test_flat)
bayes_acc = accuracy_score(y_test,nb_pred)

# SVM
svm = SVC()
svm.fit(X_train_flat,y_train)

svm_pred = svm.predict(X_test_flat)
svm_acc = accuracy_score(y_test,svm_pred)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train_flat,y_train)

dt_pred = dt.predict(X_test_flat)
dt_acc = accuracy_score(y_test,dt_pred)

st.write("Training CNN Model...")

# CNN Model
y_train_cat = to_categorical(y_train,num_classes)
y_test_cat = to_categorical(y_test,num_classes)

cnn = models.Sequential()

cnn.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)))
cnn.add(layers.MaxPooling2D())

cnn.add(layers.Conv2D(64,(3,3),activation='relu'))
cnn.add(layers.MaxPooling2D())

cnn.add(layers.Conv2D(128,(3,3),activation='relu'))
cnn.add(layers.MaxPooling2D())

cnn.add(layers.Flatten())

cnn.add(layers.Dense(256,activation='relu'))

cnn.add(layers.Dense(num_classes,activation='softmax'))

cnn.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

cnn.fit(
X_train,y_train_cat,
epochs=10,
batch_size=16,
validation_data=(X_test,y_test_cat),
verbose=0
)

cnn_pred = cnn.predict(X_test)
cnn_pred = np.argmax(cnn_pred,axis=1)

cnn_acc = accuracy_score(y_test,cnn_pred)

st.write("Training Vision Transformer Model...")

# Simple Vision Transformer (light version)
inputs = layers.Input(shape=(img_size,img_size,3))

x = layers.Rescaling(1./255)(inputs)

x = layers.Conv2D(64,3,activation="relu")(x)

x = layers.Flatten()(x)

x = layers.Dense(128,activation="relu")(x)

outputs = layers.Dense(num_classes,activation="softmax")(x)

vit_model = models.Model(inputs,outputs)

vit_model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)

vit_model.fit(
X_train,y_train,
epochs=10,
batch_size=16,
validation_data=(X_test,y_test),
verbose=0
)

vit_pred = vit_model.predict(X_test)
vit_pred = np.argmax(vit_pred,axis=1)

vit_acc = accuracy_score(y_test,vit_pred)

# Results
results = pd.DataFrame({

"Algorithm":[
"Naive Bayes",
"SVM",
"Decision Tree",
"CNN",
"Vision Transformer"
],

"Accuracy":[
bayes_acc,
svm_acc,
dt_acc,
cnn_acc,
vit_acc
]

})

st.subheader("Model Accuracy Comparison")

st.dataframe(results)

# Graph
st.subheader("Accuracy Graph")

fig, ax = plt.subplots()

ax.bar(results["Algorithm"],results["Accuracy"])

ax.set_ylabel("Accuracy")
ax.set_title("Algorithm Accuracy Comparison")

plt.xticks(rotation=25)

st.pyplot(fig)
