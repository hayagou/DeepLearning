import numpy as np

# 가중치와 바이어스

w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

def MLP(x, w, b):
    y = np.sum(w * x) + b # 가중합
    if y <= 0:
        return 0
    else:
        return 1

#NAND 게이트, 은닉층 1번째 노드
def NAND(x1, x2):
    return MLP([x1, x2], w11, b1)

#OR 게이트, 은닉층 2번째 노드
def OR(x1, x2):
    return MLP([x1, x2], w12, b2)

# AND 게이트
def AND(x1, x2):
    return MLP([x1, x2], w2, b3)

# XOR, 출력층
def XOR(x1, x2):
    return AND(NAND(x1, x2),OR(x1, x2))

if __name__ == '__main__':
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(x[0], x[1])
        print(f"입력 값 : {str(x)} 출력 값 : {str(y)}")