#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhD Code Sample (Python 3.x)
Clean ML pipeline using TensorFlow (ANN) on Kaggle Titanic dataset.

Steps:
1. Load dataset (train.csv)
2. Basic preprocessing
3. Train/test split
4. Baseline ML model (Logistic Regression)
5. ANN model (TensorFlow)
6. Evaluation and comparison

Note:
- Download Kaggle Titanic train.csv
- Place it in: data/train.csv
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

csv_path = os.path.join("data", "train.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("Please place train.csv inside the data/ folder.")

df = pd.read_csv(csv_path)

df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

X = df.drop("Survived", axis=1)
y = df["Survived"]

num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_cols = ["Sex", "Embarked"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

baseline = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=500))
])

baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, base_pred))

X_train_p = preprocess.fit_transform(X_train)
X_test_p = preprocess.transform(X_test)

if hasattr(X_train_p, "toarray"):
    X_train_p = X_train_p.toarray()
    X_test_p = X_test_p.toarray()

ann = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train_p.shape[1],)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = ann.fit(
    X_train_p, y_train,
    validation_split=0.2,
    epochs=25,
    batch_size=32,
    verbose=1
)

ann_pred = (ann.predict(X_test_p) >= 0.5).astype(int)

print("ANN Accuracy:", accuracy_score(y_test, ann_pred))
print("ANN Precision:", precision_score(y_test, ann_pred))
print("ANN Recall:", recall_score(y_test, ann_pred))
print("ANN F1:", f1_score(y_test, ann_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ann_pred))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("ANN Training Curve")
plt.show()
