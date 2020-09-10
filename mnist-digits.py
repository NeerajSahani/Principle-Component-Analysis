import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist

data = mnist.load_data()

(X_train, y_train), (X_test, y_test) = data
X_train.shape = (X_train.shape[0], -1)
X_test.shape = (X_test.shape[0], -1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(0.95)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

model.predict(X_test[0:10])
y_test[0:10]

accuracy = model.score(X_test, y_test)
