# 케라스 함수 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 라이브러리 호출
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 셋 로드
Data_set = np.loadtxt("dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:, 0:17]
Y = Data_set[:, 17]

model = Sequential()
model.add(Dense(30,input_dim=17, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 실행
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(X,Y, epochs=100)