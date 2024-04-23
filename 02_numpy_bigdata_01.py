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

# list package 확인하기
# !pip list

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f'{arr1}\n{arr2}')

# +
# 난수를 이용한 2차원 배열 생성
# randn : random + normal distribution (정규분포)

arr3 = np.random.randn(3)
arr4 = np.random.randn(2, 3)

print(f'{arr3}\n{arr4}')

# +
# 원소를 0 or 1로 초기화

arr5 = np.zeros(5)
arr6 = np.ones((2, 3))

print(f'{arr5}\n{arr6}')

# +
# arange()와 reshape()
# arange() : array + range
# arange(start, end, step)

arr7 = np.arange(20, 200, 10)
print(f'{arr7} --> {arr7.shape}')

# arr7의 개수를 먼저 확인 후 차원 변경을 한다.
arr8 = arr7.reshape(3, -1)
print(f'{arr8} --> {arr8.shape}')

# +
# 배열 다루기
ar1 = np.arange(1, 21, 1)
print(f'{ar1} --> {ar1.shape}')
print(f'index가 1인 element : {ar1[1]}')

# 차원 변경하기
ar2 = ar1.reshape(2, -1)
print(f'{ar2} --> {ar2.shape}')
print(f'index가 1인 ar2의 list : {ar2[1]}')
# -

# 배열 안의 요소 값 바꿔주기
print(ar2)
ar2[1][1] = 100
print(ar2)

# +
# 배열의 산술 연산 : 배열의 연산은 배열 원소에 각각 적용됨.

ar1 = np.arange(1, 11, 1)
ar2 = ar1 + 3 # broadcasting (각 원소에 적용됨)
ar3 = ar2 * 2

print(f'{ar1} --> {ar1.shape}\n{ar2} --> {ar2.shape}\n{ar3} --> {ar3.shape}')
# -

# 배열의 통계 메소드 사용
arr = np.array([[5, 7, 9], [-7, -6, 19], [6, 9, 11]])
print(f'{arr} --> {arr.shape}')
print(arr.sum(), arr.mean(), arr.max(), arr.min())

# axis : 배열의 축 지정
print(arr.max(axis=0))
print(arr.max(axis=1))

# 조건식을 사용한 연산
arr > 0

print(arr < 0)
print((arr < 0).sum())

# np.where : 조건
# arr < 0 : 조건식
# 조건식이 True (arr < 0) 이면 '0', 그렇지 않으면 가지고 있는 값 출력
np.where(arr < 0, 0, arr)

# +
# 배열의 정렬
arr = np.array([[5, 7, 9], [-3, -6, 19], [6, 4, 11]])
print(f'{arr} -> {arr.shape}')

# column끼리 정렬 (오름차순)
arr.sort(0)
print(arr)

# row끼리 정렬 (오름차순)
arr.sort(1)
print(arr)
# -

# ## Q&A

# +
# 2차원 -> 3차원으로 축을 추가하는 코드
# 첫 번째 : np.newaxis
# 두 번째 : np.expand_dims

# 2차원 배열 생성
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 첫 번째 방법
# 첫 번째 축([0])에 새로운 축 추가
arr_axis0 = arr[np.newaxis, :, :]
print(f'{arr_axis0} : {arr_axis0.ndim}D. --> {arr_axis0.shape}')

# 두 번째 축([1])에 새로운 축 추가
arr_axis1 = arr[:, np.newaxis, :]
print(f'{arr_axis1} : {arr_axis1.ndim}D. --> {arr_axis1.shape}')

# 세 번째 축([2])에 새로운 축 추가
arr_axis2 = arr[:, :, np.newaxis]
print(f'{arr_axis2} : {arr_axis2.ndim}D. --> {arr_axis2.shape}')

# -


