# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# ## 그래프 그리기

# +
x = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']

y1 = [15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4]
y2 = [20.1, 23.1, 23.8, 25.9, 23.4, 25.1, 26.3]

# 선 그래프 그리기
plt.plot(x, y1, label = 'seoul')
plt.plot(x, y2, label = 'busan')

# 그래프 제목
plt.title('Temperature of Cities in Korea')
# x축 이름
plt.xlabel('Day')
# y축 이름
plt.ylabel('Temperature')
# 범례 위치 지정
plt.legend(loc='upper left')

plt.show()

# +
# 위의 데이터 이용해 산점도 그리기
# 산점도 : scatterplot

plt.scatter(x, y1, label='Seoul')
plt.scatter(x, y2, label='Busan')
plt.legend()
plt.show()
# -

# 위의 데이터 이용
# bar 그래프
# plt.figure() 이용해서 여러 barplot 그리기
plt.bar(x, y1, label='Seoul')
plt.xlabel('Seoul')
plt.figure()
plt.bar(x, y2, label='Busan')
plt.xlabel('Busan')
plt.show()

# +
# barplot 겹쳐 그리기
# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(9, 4))
bar_width = 0.25

index = np.arange(7)

# 각 요일별로 서울, 부산의 bar를 순서대로 나타내기
# alpha : 막대 투명도
b1 = plt.bar(index, y1, bar_width, alpha=0.6, color='red', label='Seoul')
b2 = plt.bar(index + bar_width, y2, bar_width, alpha=0.4, color='green', label='Busan')

# x축 디자인
plt.xticks(np.arange(bar_width, 7 + bar_width, 1), x)

# 이름 설정
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.legend()

plt.show()

# +
# histogram 그리기
# 연속적인 데이터 받기
np.random.seed(1)
numbers = np.random.normal(size=10000)
print(numbers)

# 그래프 그리기
plt.hist(numbers, bins=20)

# 이름 설정
plt.xlabel('value')
plt.ylabel('freq')
plt.title('Histogram')

# 격자
plt.grid(axis='y')

plt.show()

# +
# 지수힘수
# x, y 설정
x = np.arange(0, 11)
y = 2 ** x # y = 2^x

plt.plot(x, y)
plt.xlim(0, 5)
plt.ylim(0, 40)

plt.title('y = 2^x', size=20)
plt.show()


# +
# 시그모이드 함수 만들기
def sigmoid(x):
    # 원래 함수
    s = 1 / (1 + np.exp(-x))
    # 미분 함수
    ds = s * (1 - s)
    return s, ds

x = np.arange(-10, 10, 0.1)
y1, y2 = sigmoid(x)

plt.plot(x, y1, label='sigmoid(x)')
plt.plot(x, y2, label='sigmoid_prime(x)', alpha=0.6)

# 축 설정
plt.xlim(-10, 10)
plt.axvline(0, color='gray')
plt.yticks([0, 0.5, 1])
plt.grid(axis='y')

# 제목 설정
plt.title('Sigmoid', size=20)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

plt.show()
# -

# ## Numpy 활용

# +
arr = np.arange(-15, 15)
print(f'{arr}\n--> {arr.shape}')

# reshape 사용해서 차원 변경해주기
arr = arr.reshape(5, -1)
print(f'{arr}\n--> {arr.shape}')
# -

# 인덱스를 이용해 데이터에 접근하기
print(arr[2])
print(arr[3][1])
print(arr[2, 4])
print(arr[1:, 2:5])
print(arr[:3, 1:])

# 배열의 크기
a = np.array([3, 6, 9, 12, 15, 18])
print(f'{a}\n--> {a.shape} : {a.ndim}D.')
print(a.dtype)

b = np.array([[12, 23, 34, 45],
              [56, 67, 78, 89]])
print(f'{b}\n--> {b.shape} : {b.ndim}D.')

# 배열의 유형 변경 astype()
print(b.dtype)
b = b.astype('float64')
print(f'{b}\n--> {b.dtype}')

# +
# 0으로만 이루어진 배열
print(np.zeros((3, 4)))

# 1로만 이루어진 배열 만들기
print(np.ones((3, 4)))

# 연속형 정수 생성
print(np.arange(3, 6))

# +
# 전치 행렬 : 행과 열을 바꾼다.
a = np.ones((3, 4))
print(a)

b = np.transpose(a)
print(b)

# +
# 배열의 사칙 연산
a = np.array([[2, 3, 4],
              [6, 7, 8]])
b = np.array([[10, 11, 12],
              [13, 14, 15]])
print(a)
print(b)

# 배열의 덧셈
print(a + b)
# 배열의 곱셈
print(a * b)
# 배열의 나눗셈
print(b / a)
# -

# 크기가 서로 다른 배열끼리 더하기
c = np.array([121, 144, 169])
print(c + 4)
print(a + c) # 분배법칙이 사용됨.

# +
# 파이썬 리스트와 넘파이 배열의 차이점
f = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15]])
print(f'{f}\n--> {type(f)}')

f_list = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15]]
print(f'{f_list}\n--> {type(f_list)}')
# -

f_list[0] = 100
print(f_list)

f[:2] = 100
print(f)


