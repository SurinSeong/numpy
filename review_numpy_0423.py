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

# 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f'{arr1}\n{arr2}')

# 난수를 이용한 2차원 배열 생성
# randn : random + normal distribution (정규분포)
arr3 = np.random.randn(3)
arr4 = np.random.randn(2, 3)
print(f'< arr3 >\n{arr3}\n< arr4 >\n{arr4}')

# 원소를 0 또는 1로 초기화 한다.
arr5 = np.zeros(5)
arr6 = np.ones((2, 3))
print(f'{arr5}\n{arr6}')

# +
# arange()와 reshape()
# arange() : array + range
# arange(start, end, step)
arr7 = np.arange(20, 200, 10)
print(f'{arr7}\n{arr7.shape}')

# arr7의 개수를 먼저 확인 후 차원 변경
arr8 = arr7.reshape(3, -1)
print(f'{arr8}\n{arr8.shape}')

# +
# 배열 다루기
arr1 = np.arange(1, 21, 1)
print(f'{arr1} \n--> {arr1.shape}')
print(f'index가 1인 element : {arr1[1]}')

# 차원 변경하기
arr2 = arr1.reshape(2, -1)
print(f'{arr2}\n--> {arr2.shape}')
print(f'arr2의 index가 1인 행과 0인 열의 element : {arr2[1][0]}')

# +
# 배열 안의 요소 값 바꿔주기
print(arr2)
print('▽')

arr2[1][5] = -1
print(arr2)

# +
# 배열의 산술 연산 : 배열의 연산은 배열 원소에 각각 적용된다.
arr1 = np.arange(1, 11, 1)
# broadcasting (각 원소에 적용된다.)
arr2 = arr1 + 3
arr3 = arr2 * 2

print(arr1)
print(f'arr1의 모든 요소에 3을 더함.\n--> {arr2}')
print(f'arr2의 모든 요소에 2를 곱함.\n--> {arr3}')
# -

# 배열의 통계 메소드 사용
arr = np.array([[5, 7, 9],
                [-1, 10, 3],
                [-3, -5, 11]])
print(f'{arr}\n--> {arr.shape}')
print(f'요소끼리의 합 : {arr.sum()}\n요소의 평균 : {arr.mean()}\n최댓값 : {arr.max()}\n최솟값 : {arr.min()}')

# axis : 배열의 축 지정
print(f'column 기준 최댓값 : {arr.max(axis=0)}')
print(f'row 기준 최댓값 : {arr.max(axis=1)}')

# 조건식을 이용한 연산
condition = (arr > 0)
print(condition)
print(f'조건이 참인 요소의 개수 : {condition.sum()}')

# 배열의 정렬
print(arr)
# column끼리 정렬 (오름차순)
arr.sort(0)
print(arr)
# row끼리 정렬 (오름차순)
arr.sort(1)
print(arr)

# 2차원 -> 3차원
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
arr_axis0 = arr[np.newaxis, :, :]
print(f'{arr_axis0} : {arr_axis0.ndim}D.\n --> {arr_axis0.shape}')
arr_axis1 = arr[:, np.newaxis, :]
print(f'{arr_axis1} : {arr_axis1.ndim}D.\n --> {arr_axis1.shape}')
arr_axis2 = arr[:, :, np.newaxis]
print(f'{arr_axis2} : {arr_axis2.ndim}D.\n --> {arr_axis2.shape}')

# np.zeros_like()
one_a = np.ones((3, 4))
print(one_a)
print(np.zeros_like(one_a))

one_b = np.array([[1, 2, 3],
                  [4, 5, 6]])
print(one_b)
print(np.zeros_like(one_b))

# np.full()
print(np.full((4, 3), 9))

# +
# seed 생성
np.random.seed(1)

# random : 0 ~ 1 중 무작위 수
# 배열의 형태(파라미터)를 튜플 형태로 받음
print(np.random.random((3, 4)))

# rand : 0 ~ 1 중 무작위 수
# 여러 개의 파라미터를 받아 차원 설정 가능
print(np.random.rand(3, 4))
print(np.random.rand(2, 3, 4))

# randn : 표준정규분포
print(np.random.randn(3, 4))

# +
# np.eye() : 항등 행렬
# A = I * A
I = np.eye(4)

A = np.array([[18, 11, 25, 4],
              [1, 7, 3, 9],
              [10, 54, 29, 30],
              [2, 8, 5, 4]])
print(f'"항등행렬"\n{I}')
print(f'"4x4 행렬"\n{A}')

# 항등 행렬과의 내적은 자기 자신이다.
print(A @ I)

# +
# numpy array 크기
arr = np.arange(12)
print(f'{arr}\n--> {arr.shape}')

arr2 = arr.reshape(3, -1)
print(f'{arr2}\n--> {arr2.shape}')
# -

# order='F' : 값을 수직으로 채워넣음.
print(arr.reshape(4, -1, order='F'))

# +
print(arr)

# 3차원으로 reshape
arr4 = arr.reshape(2, 3, 2)
print(f'{arr4}\n--> {arr4.ndim}D. {arr4.shape}')

# order 옵션 사용
arr5 = arr.reshape(2, 3, 2, order='F')
print(f'{arr5}\n--> {arr5.ndim}D.')

# 1차원으로 변환
print(f'{arr4}\n{arr4.flatten()}')
# -

# 인덱싱, 슬라이싱
arr1 = np.arange(1, 11)
print(arr1)
print(arr1[2], arr1[-5])

arr2 = arr1.reshape(5, -1)
print(f'{arr2}\n--> {arr2.ndim}D')
print(arr2[3, 1])
print(arr2[-3, -2])
print(arr2[-2:, :-1])

# +
# 내적
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr2 = np.array([[7, 8],
                 [9, 10],
                 [11, 12]])
print(f'{arr1}\n&\n{arr2}')
print(f'{arr1@arr2}\n==> arr1과 arr2의 내적')
print(np.dot(arr1, arr2))

# 전치 행렬
print(f'arr1의 전치 행렬\n{arr1.T}')
print(f'arr1과 이것의 전치 행렬의 내적\n{arr1@arr1.T}')
print(np.transpose(arr1))
# -

# ## 기술통계

# +
np.random.seed(10)

x = np.random.randint(-100, 100, (10, ))
print(x)
print(len(x))
print('MEAN :', np.mean(x))
print('VAR :', np.var(x))
print('STD :', np.std(x))
print('MAX :', np.max(x))
print('MIN :', np.min(x))
print('MEDIAN :', np.median(x))
# -

x.sort()
print(x)

# 사분위수
Q1 = np.percentile(x, 25)
Q2 = np.percentile(x, 50)
Q3 = np.percentile(x, 75)
print('Q1 :', Q1)
print('Q2 :', Q2)
print('Q3 :', Q3)
print('IQR :', Q3 - Q1)
print(f'Outlier 범위\n~{Q1}, {Q3}~')






