import numpy as np

# 기울기 a와 y 절편 b, 최소 제곱법으로 구해진 값들
fake_a_b = [3, 76]

# x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y= ax + b 에 a 와 b 값을 대입하여 결과를 출력하는 함수
def predict(x):
    return fake_a_b[0]*x + fake_a_b[1]

# MSE 함수
def mse(y_hat, y):
    return ((y_hat-y) ** 2).mean()

# MSE 함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def mse_val(predict_result, y):
    return mse(np.array(predict_result), np.array(y))

# 예측 값이 들어갈 빈 리스트
predict_result = []

# 모든 x 값을 한 번 씩 대입하여
for i in range(len(x)):
    # 그 결과에 해당하는 predict_result 리스트를 완성
    predict_result.append(predict(x[i]))
    print(f"공부시간={x[i]}, 실제 점수={y[i]}, 예측 점수={predict(x[i])}")

# 최종 MSE 출력
print(f"MSE 최종값: {str(mse_val(predict_result,y))}")

# ------------My MSE-----------

# 예측 값 계산
# Input : X data
# Output : 최소제곱법으로 구해진 1차식의 Y 값
def val_Y(x):
    return fake_a_b[0]*x+fake_a_b[1]

# 오차 계산
# Input : Y Date, Y 예측을 위한 X값
# Output : 예측값과 실제값의 오차를 반환
def val_Error(x, y):
    return y-val_Y(x)

error_list = [] # 계산된 오차를 저장하기 위한 변수

# 오차 계산을 위한 반복
for i in range(len(x)):
    error_list.append( val_Error(x[i], y[i])**2 ) # 오차를 구한뒤 제곱 후 리스트 추가

_mse = sum(error_list)/len(x) # 오차의 제곱을 모두 더한뒤 원소의 갯수로 나눔

print(f"My MSE 최종값 : {_mse}")

# 예제와 직접 코딩한 MSE의 결과가 같음, 책의 예제는 np의 사용법을 모르면 직관적으로 이해하기가 힘듦...