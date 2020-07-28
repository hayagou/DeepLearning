import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부 시간 X와 합격 여부 Y의 리스트 만들기
data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]

x_data = [i[0] for i in data]
y_data = [i[1] for i in data]


# 그래프로 나타내기
plt.scatter(x_data,y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)

# 기울기 a와 절편 b의 값 초기화
a = 0
b = 0

# 학습률
lr = 0.05

# 시그모이드 함수 정의
def sigmoid(x):
    return 1/ (1 + np.e ** -x)

# 경사 하강법 실행
# 1000번 반복될 때마다 각 x_data 값에 대한 현재의 a 값, b 값 출력
for i in range(2001):
    for x_data, y_data in data:
        # a에 관해 편미분
        a_diff = x_data*(sigmoid(a*x_data +b) - y_data)
        # b에 관한 편미분
        b_diff = sigmoid(a*x_data+ b) - y_data
        a -= lr * a_diff
        b -= lr * b_diff
        if i % 1000 == 0:
            print(f"epoch={i}, 기울기={round(a, 4)}, 절편={round(b, 4)}")

# 앞서 구한 기울기와 절편을 이용해 그래프를 그려 봅니다.
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 15, 0.1)) #그래프로 나타낼 x값의 범위를 정합니다.
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a*x + b) for x in x_range]))
plt.show()