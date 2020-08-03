from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

# seed 값 설정
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 입력
df = pd.read_csv("dataset/sonar.csv", header=None)

dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

# 문자열 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=200, batch_size=5)

print(f"Accuracy : {model.evaluate(X, Y)[1]}")