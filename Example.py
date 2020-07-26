from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

Data_set = np.loadtxt("C:/Users/HJ/Downloads/lung_cancer.csv", delimiter=",")

X = Data_set[:, 0:4]
Y = Data_set[:, 4]

model = Sequential()
model.add(Dense(30, input_dim = 4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, epochs=100, batch_size=10)

