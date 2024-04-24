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

# +
x = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']

y1 = [15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4]
y2 = [20.1, 23.1, 23.8, 25.9, 23.4, 25.1, 26.3]

# 선그래프 그리기 -> x축에 해당하는 데이터, y축에 해당하는 데이터, 이름
plt.plot(x, y1, label='seoul')
plt.plot(x, y2, label='busan')

# 그래프 제목
plt.title('Temperature of cities in Korea')
# x축 이름
plt.xlabel('Day')
# y축 이름
plt.ylabel('Temperature')
# 범례 위치 지정
plt.legend(loc='upper left')


# +
# 산점도 (Scatter plot) : 흩어진 정도
#  >> 관측한 실데이터

x = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']

plt.scatter(x, [15, 14, 16, 18, 17, 20, 22])
plt.show()
# -

# * 막대 그래프
#     * x 데이터 - 범주형 데이터
#
# * 히스토그램
#     * x 데이터 - 연속형 데이터

# +
# bar 그래프 : 범주형 그래프
x = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
y = [15, 14, 16, 18, 17, 20, 22]

plt.bar(x, y)
plt.show()

# +
# 히스토그램 --> 정규 분포와 연결됨.

# 연속적인 데이터 받기
np.random.seed(1)
numbers = np.random.normal(size=10000)
print(numbers)

# 그래프 그리기
plt.hist(numbers, bins=20) # bins : 계급 나누기
# x축 이름
plt.xlabel('value')
# y축 이름
plt.ylabel('freq')
# 격자 표시
plt.grid(axis='y')
plt.show()

# +
# 무작위 데이터 받기
np.random.seed(1)
data = np.random.randn(1000)

# 히스토그램
plt.hist(data, bins=20)

plt.title('HISTOGRAM')
plt.xlabel('value')
plt.ylabel('freq')
# 격자 넣기
plt.grid(True)
plt.show()

# +
# 지수함수

# x, y
x = np.arange(0, 10)
y = x ** 2

plt.plot(x, y) # y  = x^2 함수
plt.show()

# +
x = np.arange(10)
y1 = np.ones(10)
y2 = x
y3 = x ** 2

# plt.plot(x, y1, x, y2, x, y3) : 이것도 가능
plt.plot(x, y1, label='y=1')
plt.plot(x, y2, label='y=x')
plt.plot(x, y3, label='y=x^2')
plt.legend()
plt.show()


# +
# 시그모이드 함수 만들기
def sigmoid(x):
    # 시그모이드 함수
    s = 1 / (1 + np.exp(-x))
    # s를 미분한 값
    ds = s * (1 - s)
    return s, ds

x = np.arange(-10, 10, 0.1)
y1, y2 = sigmoid(x)

# plt.plot(x, y1, x, y2) : 이것도 가능
plt.plot(x, y1, label='sigmoid(x)')
plt.plot(x, y2, label='sigmoid_prime(x)')

# x축 최대 최소 설정
plt.xlim(-10, 10)
# x축 설정
plt.axvline(0, color='k')
# y축 설정
plt.yticks([0, 0.5, 1])
# y축 그리드 설정
plt.grid(axis='y')

plt.title('Sigmoid', fontsize=20)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()
plt.show()
# -

# 배열 슬라이싱
arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15]])
print(arr)
print(arr[1])
print(arr[1][2])
# 행렬구조에 바로 접근이 가능하다 (numpy의 장점)
print(arr[1, 2])
print(arr[1:, 3:])
print(arr[:2, 2:])

# +
# 배열의 크기
a = np.array([2, 3, 4, 5, 6])
print(f'{a} --> {a.shape} : {a.ndim}D')

# 배열의 원소 유형
print(a.dtype)
# -

b = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])
print(f'{b}\n--> {b.shape} : {b.ndim}D')


# +
# 배열의 유형 변경 astype()
data = np.arange(1, 5)
print(f'{data}\n--> {data.dtype}') # 데이터 타입이 정수라는 것을 알 수 있음.

# 정수 -> 실수 (타입변경 전 미리 출력시켜보기)
data = data.astype('float64')
print(f'{data}\n--> {data.dtype}')

# +
# int(정수), unit(unsigned(기호(+, -) 없음) integer), float(실수)
# complex : 복소수 (실수 + 허수 : 3+4i, i^2 = -1)
# bool (참/거짓), string(문자열), object(객체)

# +
# 0으로 이루어진 배열 만들기
print(np.zeros((2, 5)))

# 1로 이루어진 배열 만들기
print(np.ones((2, 5)))

# 연속형 정수 생성하기
print(np.arange(2, 5))
# -

# 전치 행렬 : 행과 열 바꾸기
a = np.ones((2, 3))
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
# 배열의 덧셈 : 원소끼리 더해짐
print(a+b)
# 배열의 곱셈
print(a*b)
# 배열의 나눗셈
print(a/b)
# -

# 크기가 서로 다른 배열끼리 더하기
# Broadcasting
c = np.array([100, 200, 300])
print(c + 3)
print(a + c) # 분배법칙이 사용됨.

# broadcasting이 되지 않는 경우 : 
d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f'{d}\n--> {d.shape}')
print(f'{a}\n--> {a.shape}')
# print(a + d) # 행과 열의 크기가 모두 다른 배열은 브로드 캐스팅해서 더할 수 없음.

e = np.array([[9],
              [3]])
print(e)
print(e.shape)
print(a)
print(a.shape)
print(a + e) # 행의 개수가 같음.

# +
# 파이썬 리스트와 넘파이 배열의 차이점
f = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15]])
print(f)
print(type(f))

f_list = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15]]
print(f_list)
print(type(f_list))
# -

print(f_list)
f_list[0] = 0
print(f_list)

print(f)
f[:2] = 0 # numpy는 원소 자체에 접근 가능
print(f)

# ## 인덱싱, 슬라이싱

 arr = np.arange(10)
 print(arr)
 print(arr[:5])
 print(arr[-4:])

a = np.arange(2, 8).reshape(2, 3)
print(a[1, 2])
print(a[:, 2])
