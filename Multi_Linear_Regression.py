import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부 시간 X와 성적 Y의 리스트 만들기
data = [[2, 0, 1, 81], [4, 4, 2, 93], [6, 2, 3, 91], [8, 3, 4, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
x3 = [i[2] for i in data]
y = [i[3] for i in data]

# 그래프로 확인
# ax = plt.axes(projection="3d")
# ax.set_xlabel("study_hours")
# ax.set_ylabel("private_class")
# ax.set_zlabel("Score")
# ax.dist = 11
# ax.scatter(x1, x2, y)
# plt.show()

# 리스트로 되어 있는 x와 y 값을 넘파이 배열로 바꾸기(인덱스로 하나씩 불러와 계산할 수 있도록 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
x3_data = np.array(x3)
y_data = np.array(y)

# 기울기 a와 절편 b의 값 초기화
a1 = 0
a2 = 0
a3 = 0
b = 0

# 학습률
lr = 0.003

# 몇 번 반복할지 설정(0부터 세므로 원하는 반복 횟수에 +1)
epochs = 20001

# 경사 하강법 시작
for i in range(epochs): # epoch 수 만큼 반복
    y_predict = a1 * x1_data + a2 * x2_data + a3 * x3_data + b # Y를 구하는 식 세우기
    error = y_data - y_predict # 오차를 구하는 식
    # 오차 함수를 a1으로 미분한 값
    a1_diff = -(1/len(x1_data)) * sum(x1_data * error)
    # 오차 함수를 a2로 미분한 값
    a2_diff = -(1/len(x2_data)) * sum(x2_data * error)

    a3_diff = -(1/len(x3_data)) * sum(x3_data * error)
    # 오차 함수를 b로 미분한 값
    b_diff = -(1/len(x1_data)) * sum(y_data-y_predict)
    a1 -= lr * a1_diff
    a2 -= lr * a2_diff
    a3 -= lr * a3_diff
    b -= lr * b_diff

    if i % 100 == 0:
        print(f"epoch= {i}, 기울기1={round(a1, 4)}, 기울기2={round(a2, 4)}, 기울기3={round(a3, 4)}, 절편={round(b, 4)}")

# #참고 자료, 다중 선형회귀 '예측 평면' 3D로 보기
#
# import statsmodels.api as statm
# import statsmodels.formula.api as statfa
# #from matplotlib.pyplot import figure
#
# X = [i[0:2] for i in data]
# y = [i[2] for i in data]
#
# X_1=statm.add_constant(X)
# results=statm.OLS(y,X_1).fit()
#
# hour_class=pd.DataFrame(X,columns=['study_hours','private_class'])
# hour_class['Score']=pd.Series(y)
#
# model = statfa.ols(formula='Score ~ study_hours + private_class', data=hour_class)
#
# results_formula = model.fit()
#
# a, b = np.meshgrid(np.linspace(hour_class.study_hours.min(),hour_class.study_hours.max(),100),
#                    np.linspace(hour_class.private_class.min(),hour_class.private_class.max(),100))
#
# X_ax = pd.DataFrame({'study_hours': a.ravel(), 'private_class': b.ravel()})
# fittedY=results_formula.predict(exog=X_ax)
#
# fig = plt.figure()
# graph = fig.add_subplot(111, projection='3d')
#
# graph.scatter(hour_class['study_hours'],hour_class['private_class'],hour_class['Score'],
#               c='blue',marker='o', alpha=1)
# graph.plot_surface(a,b,fittedY.values.reshape(a.shape),
#                    rstride=1, cstride=1, color='none', alpha=0.4)
# graph.set_xlabel('study hours')
# graph.set_ylabel('private class')
# graph.set_zlabel('Score')
# graph.dist = 11
#
# plt.show()