import numpy as np

# 최소 제곱법 구현

#data
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

#x와 y의 평균
mx = np.mean(x)
my = np.mean(y)

print(f"x의 평균값:{mx}")
print(f"y의 평균값:{my}")

#분모
divisor = sum([(i-mx) ** 2 for i in x])

#분자
def top( x, mx, y, my ):
    d = 0
    for i in range(len(x)):
        d = d + (x[i]-mx)*(y[i]-my)
    return d

dividend = top(x, mx, y, my)

print(f"분모: {divisor}")
print(f"분자: {dividend}")

# 기울기와 y 절편 구하기
a = dividend / divisor
b = my - (mx*a)

print(f"기울기 : {a}")
print(f"y 절편 : {b}")
