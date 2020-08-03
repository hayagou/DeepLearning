from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("dataset/sonar.csv", header=None)

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]


e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy= []

for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    model.fit(X[train], Y[train], epochs=100, batch_size=5)
    k_accuracy = model.evaluate(X[test],Y[test])[1]
    accuracy.append(k_accuracy)

print(accuracy)