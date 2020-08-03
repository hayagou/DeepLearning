from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# seed값 설정
np.random.seed(3)
tf.random.set_seed(3)

df = pd.read_csv("dataset/iris.csv", names= ["sepal_length", "sepal_width", "petal_lenght", "petal_width", "species"])

# 그래프로 확인
sns.pairplot(df, hue="species")
plt.show()

# 데이터 분류
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

model = Sequential()
model.add(Dense(16, input_dim=4, activation="relu"))
model.add(Dense(3, activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
model.fit(X, Y_encoded, epochs=50, batch_size=1)

print(f"Acuuracy : {round(model.evaluate(X, Y_encoded)[1], 4)}")