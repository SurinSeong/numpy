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

arr1 = np.array([1, 2, 3])
print(f'{arr1}, {type(arr1)}, {arr1.shape}')

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f'{arr2}, {type(arr2)}, {arr2.shape}')

arr3 = np.array([[1, 2, 3]])
print(f'{arr3}, {type(arr3)}, {arr3.shape}, {arr3.ndim}D.')

# 차원 확인
print(f'arr1 : {arr1.ndim}D.')
print(f'arr2 : {arr2.ndim}D.')
print(f'arr3 : {arr3.ndim}D.')

# arange : array + range
print(np.arange(3))
print(np.arange(1, 20, 3))

# numpy array 초기화
np.zeros((2, 5))

# * np.zeros_like() : () 안의 배열과 동일한 모양을 가진 배열을 0으로 채워서 생성하기

one_a = np.ones((3, 4))
print(one_a)
print(np.zeros_like(one_a))

one_b = np.array([[1, 2, 3], [4, 5, 6]])
print(one_b)
print(np.zeros_like(one_b))

# np.full()
print(np.full((4, 3), 9))

# +
# np.random.random() vs. np.random.rand()
# seed 생성
np.random.seed(1)

# random : 0 ~ 1 중 무작위 수
# 배열의 형태(파라미터)를 튜플 형태로 받는다.
print(np.random.random((3, 4)))

# rand : 0 ~ 1 중 무작위 수
# 여러 개의 파라미터를 받아 차원을 설정할 수 있다.
print(np.random.rand(3, 4))
print(np.random.rand(2, 3, 4))

# randn : 표준정규분포
print(np.random.randn(3, 4))
# 표준화 : (x - x_Bar) / std
# -

# np.eye() : 항등 행렬
# A * I = A
print(np.eye(4))

# +
# 일반 행렬과 항등 행렬과의 내적은 자기 자신이다.

A = np.array([[2, 3, 1, 5],
              [8, 1, 3, 4],
              [5, 6, 7, 8],
              [1, 2, 3, 4]
              ])
print(A)

I = np.eye(4)
print(I)

print(A@I)

# +
# numpy array 크기

arr = np.arange(12)
print(f'{arr} --> {arr.shape}')
arr2 = arr.reshape(3, -1)
print(f'{arr2} --> {arr2.shape}')
# -

# order = 'F' : 값을 수직으로 채워넣음.
arr.reshape(4, -1, order='F')

# +
arr1 = np.arange(10)
print(arr1)

arr2 = arr1.reshape(-1, 5)
print(f'{arr2} --> {arr2.ndim}')

# +
arr3 = np.arange(12)
print(f'{arr3} --> {arr3.shape}')

arr4 = arr3.reshape(2, 3, 2)
print(f'{arr4} --> {arr4.ndim}. {arr4.shape}')
# -

arr5 = arr3.reshape(2, 3, 2, order='F')
print(f'{arr5} --> {arr5.ndim}.')

# 1차원으로 변환
print(f'{arr2}\n{arr2.flatten()}')

# numpy array 데이터 추출
# indexing
arr1 = np.arange(1, 10)
print(arr1)
print(arr1[2], arr1[-2])

arr2 = arr1.reshape(3, -1)
print(f'{arr2} --> {arr2.ndim}')
print(arr2[0, 0])
print(arr2[-1, -2])
print(arr2[-1, :-1])

# +
print(arr1)

arr2 = arr1[0:3]
print(arr2)

arr3 = arr2[1:]
print(arr3)

arr4 = arr1.reshape(3, -1)
print(arr4)
# -

# 슬라이싱
print(arr4)
print(arr4[:2, :2])
print(arr4[1:3, :3])
print(arr4[:2, 1:])


# +
# 내적
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8], [9, 10], [1, 2]])

print(f'{arr1}\n{arr2}')
print(f'{arr1 @ arr2} --> 내적')
print(np.dot(arr1, arr2))

# 전치 행렬
print(f'\n{arr1}\n⊙\n{arr1.T}\n내적\n{arr1@arr1.T}')
print(arr1)
print(np.transpose(arr1))
# -

# ## numpy를 이용한 기술통계

x = np.array([18, 26, 56, 9, 76, 34, -2])
print(len(x))
# 평균
print('MEAN :', np.mean(x))
# 분산
print('VARIATION :', np.var(x))
# 표준편차
print('STANDARD DEVIATION :', np.std(x))

print(x)
print('MAX :', np.max(x))
print('MIN :', np.min(x))
print('MEDIAN :', np.median(x))
x.sort()
print(x)

# +
# 사분위수 --> boxplot에서 사용됨.

print('Q1 :', np.percentile(x, 25))
# Q2는 중앙값과 같다.
print('Q2 :', np.percentile(x, 50))
print('Q3 :', np.percentile(x, 75))
print('IQR :', np.percentile(x, 75) - np.percentile(x, 25))
print('OUTLIER :', np.percentile(x, 25), '보다 작고', np.percentile(x, 75), '보다 크다.')
# -


