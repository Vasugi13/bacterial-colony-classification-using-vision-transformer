# =========================================
# 1. Mount Google Drive
# =========================================
from google.colab import drive
drive.mount('/content/drive')

# =========================================
# 2. Import Libraries
# =========================================
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

# =========================================
# 3. Dataset Path
# =========================================
dataset_path = "/content/drive/MyDrive/bacterial colony dataset"

# =========================================
# 4. Selected Classes (5 classes from 31)
# =========================================
selected_classes = [
"Escherichia.coli",
"Staphylococcus.aureus",
"Candida.albicans",
"Lactobacillus.plantarum",
"Enterococcus.faecalis"
]

# =========================================
# 5. Load Images
# =========================================
images = []
labels = []

img_size = 128

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

print("Dataset Shape:",X.shape)

# =========================================
# 6. Encode Labels
# =========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(np.unique(y_encoded))

# =========================================
# 7. Train Test Split
# =========================================
X_train,X_test,y_train,y_test = train_test_split(
X,y_encoded,test_size=0.2,random_state=42
)

# =========================================
# 8. Flatten Data for ML Models
# =========================================
X_train_flat = X_train.reshape(X_train.shape[0],-1)
X_test_flat = X_test.reshape(X_test.shape[0],-1)

# =========================================
# 9. Naive Bayes
# =========================================
nb = GaussianNB()
nb.fit(X_train_flat,y_train)

nb_pred = nb.predict(X_test_flat)

bayes_acc = accuracy_score(y_test,nb_pred)

print("Naive Bayes Accuracy:",bayes_acc)

# =========================================
# 10. SVM
# =========================================
svm = SVC()
svm.fit(X_train_flat,y_train)

svm_pred = svm.predict(X_test_flat)

svm_acc = accuracy_score(y_test,svm_pred)

print("SVM Accuracy:",svm_acc)

# =========================================
# 11. Decision Tree
# =========================================
dt = DecisionTreeClassifier(max_depth=3)

dt.fit(X_train_flat,y_train)

dt_pred = dt.predict(X_test_flat)

dt_acc = accuracy_score(y_test,dt_pred)

print("Decision Tree Accuracy:",dt_acc)

# =========================================
# 12. CNN Model
# =========================================
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
validation_data=(X_test,y_test_cat)
)

cnn_pred = cnn.predict(X_test)
cnn_pred = np.argmax(cnn_pred,axis=1)

cnn_acc = accuracy_score(y_test,cnn_pred)

print("CNN Accuracy:",cnn_acc)

# =========================================
# 13. Simple Vision Transformer
# =========================================
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
validation_data=(X_test,y_test)
)

vit_pred = vit_model.predict(X_test)

vit_pred = np.argmax(vit_pred,axis=1)

vit_acc = accuracy_score(y_test,vit_pred)

print("Vision Transformer Accuracy:",vit_acc)

# =========================================
# 14. Model Accuracy Comparison
# =========================================
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

print("\nMODEL ACCURACY COMPARISON\n")

print(results)

# =========================================
# 15. Accuracy Graph
# =========================================
plt.figure(figsize=(8,5))

plt.bar(results["Algorithm"],results["Accuracy"])

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=25)

plt.show()
