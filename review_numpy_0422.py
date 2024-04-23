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

# 사용할 클래스 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# +
# list와 numpy의 차이
# list
list1 = [1, 2, 3]
print('list :', list1)

# array
arr = np.array([1, 2, 3])
print('array :', arr) # 리스트와 같이 출력됨 (하지만 ,가 없음)
print(arr[0]) # 인덱싱 가능 = 순서가 있음.
print(arr[-1])

# +
# numpy 배우는 이유
# list + list = 연장된 list
a = [1, 2, 3]
b = [4, 5, 6]
c = a + b
print(c, len(c))

# arrat + array = 같은 위치끼리 더해짐
arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])
arr_c = arr_a + arr_b
print(arr_c, len(c))
# -

# 여러 개의 리스트를 담으려면 []안에 여러 list를 작성해야 한다.
arr_d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('arr_d :', arr_d)
print('last list :', arr_d[-1])
print('last element :', arr_d[-1][-1])

# ## Numpy 배열 속성

# shape : 배열의 형태 확인하기 ==> (행, 열)
# 무조건 확인해야 함!!
print(arr_d, arr_d.shape)

# 1차원
print(arr_a, arr_a.shape)

arr_e = np.array([[1], [2], [3]])
print(arr_e, arr_e.shape)

arr_f = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_f, arr_f.shape)

# 배열의 차원 : ndim(number of dimension)
print('arr_a의 차원 :', arr_a.ndim, 'arr_f의 차원 :', arr_f.ndim)

# dtype(data type) : 요소의 자료형
arr_f.dtype

# itemsize : 요소 하나의 크기
arr_f.itemsize

# size : 전체 요소의 개수
arr_f.size

print(np.zeros((3, 4)))
print(np.ones((3, 4)))
print(np.ones((3, 4)).dtype)

# 항등행렬
print(np.eye(3))

# arange() (array + range) : 배열의 범위
# np.arange(start, stop, step)
print(np.arange(5))
print(np.arange(1, 6))
print(np.arange(1, 10, 2))

# linspace()
# 0 ~ 10에서 동등하게 10개로 균등하게 나누기
print(np.linspace(0, 10, 10))
print(np.linspace(1, 10, 5))

# ## 배열 합치기

# +
# 2개의 배열 합치기
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print('x :', x, '\ny :', y)
# -

# 함수 이용해서 배열 합치기
# concatenate() : 배열 합치기 (default : axis=0)
concat_0 = np.concatenate((x, y))
concat_1 = np.concatenate((x, y), axis=1)
print('row를 기준으로\n', concat_0, '\n', concat_0.shape)
print('column을 기준으로\n', concat_1, '\n', concat_1.shape)

# +
# vstack : 수직으로 쌓기
# == concatenate(axis=0)
# hstack : 수평으로 쌓기
# == concatenate(axis=1)

print('수직으로 쌓으면\n', np.vstack((x, y)))
print('수평으로 쌓으면\n', np.hstack((x, y)))
# -

# ## reshape

# reshape --> 차원 변화, 형태 변화
d = np.arange(12)
print(f'{d} is {d.ndim}D, {d.shape}.')

# 1차원에서 2차원으로
dim_2d = d.reshape(3, 4)
print(f'{dim_2d} is {dim_2d.ndim}D, {dim_2d.shape}.')

# 1차원에서 2차원으로 변경할 때, row의 개수와 column의 개수를 지정해야 한다.
'''
하지만 데이터의 개수가 너무 많을 경우 직접 정할 수 없기 때문에,
row나 column의 개수를 정한 후, 나머지를 -1로 정하면 된다.
'''
dim_2d_02 = d.reshape(3, -1)
print(f'{dim_2d_02} : column을 -1로 정함.')

# +
# row나 column을 -1로 정하기
temp = np.arange(120)
print(f'{temp} : {temp.ndim}D. --> {temp.shape}')

temp_2d = temp.reshape(-1, 3)
print(f'{temp_2d} : {temp_2d.ndim}D. --> {temp_2d.shape}')
# -

# 이때 row와 column의 개수는 전체 데이터 개수의 약수이어야 한다.
temp2 = np.arange(11)
print(f'{temp2} : {temp2.ndim}D. --> {temp2.shape}')

# 위의 temp2의 약수는 1과 11뿐이기 때문에 이외의 수로 row와 column수를 정할 수 없다.
# temp2.reshape(2, -1)
'''
ValueError: cannot reshape array of size 11 into shape (2,newaxis)
'''

# ## 배열 분할

# +
arr_g = np.arange(30)
print(f'{arr_g} : {arr_g.ndim}D. --> {arr_g.shape}')

arr_g_2d = arr_g.reshape(-1, 10)
print(f'{arr_g_2d} : {arr_g_2d.ndim}D. --> {arr_g_2d.shape}')
# -

# 하나의 array를 분할한 후 따로 저장가능하다.
arr_g_01, arr_g_02 = np.split(arr_g_2d, [4], axis=1) # 중간에는 인덱스, axis=0는 row, axis=1는 column
print(f'{arr_g_01} <-- FIRST,\n {arr_g_02} <-- SECOND')

# newaxis
arr_h = np.arange(1, 7)
print(f'{arr_h} <-- {arr_h.ndim}D. : {arr_h.shape}')

arr_h_2d = arr_h[np.newaxis, :]
print(f'{arr_h_2d} : {arr_h_2d.ndim}D. --> {arr_h_2d.shape}')

arr_h2_2d = arr_h[:, np.newaxis]
print(f'{arr_h2_2d} : {arr_h2_2d.ndim}D. --> {arr_h2_2d.shape}')

# ## 인덱싱과 슬라이싱

ages = np.array([26, 28, 19, 17, 33])
print(f'{ages} : {ages.ndim}D. --> {ages.shape}')

print(f'index가 2인 값 : {ages[2]}')
print(f'slicing : {ages[1:3]}, {ages[:-2]}')

# +
# 논리적 인덱싱 --> 조건을 이용하기
condition = (ages > 20)

# 조건이 True인 것만 뽑기
print(f'조건에 맞는 값 : {ages[condition]}')
# -

# 2차원 배열 인덱싱하기
a_2d = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(f'{a_2d} : {a_2d.ndim}D. --> {a_2d.shape}')
print(f'The first list : {a_2d[0]},\nThe third element of the first list : {a_2d[0][2]}')
print(f'row index : 0, column index : 2 --> {a_2d[0, 2]}')

# array안의 element 수정 가능하다.
print(a_2d)
a_2d[0, 0] = 12
print(a_2d)

# 인덱싱으로 원하는 행만 뽑기
print(f'{a_2d[0:2]} : a_2d의 두 번째 행까지 출력. {a_2d[0:2].shape}')

print(f'{a_2d[0:2, 1:3]} : a_2d의 첫 번째부터 두 번째 행까지와 두 번째 열부터 세 번째 열까지')

# 실무에서 사용한다던데
print(a_2d[::2, ::2])
print('첫 번째 행의 첫 번째 요소, 세 번째 요소\n세 번째 행의 첫 번째 요소, 세 번째 요소')

# ## numpy 연산

arr1 = np.array([[1, 2], [3, 4], [5, 6]])
arr2 = np.array([[1, 1], [1, 1], [1, 1]])
print(arr1, arr2)

result = arr1 + arr2
print(result)

# Broad Casting
np.array([1, 2, 3])

miles = np.array([1, 2, 3])
print(miles)
print(miles * 1.6)
print(miles[0] * 10)

# numpy 곱셈
arr3 = np.array([[2, 2], [2, 2], [2, 2]])
result = arr1 * arr3
print(f'{arr1}\n*\n{arr3}\n=\n{result}')

# +
# 내적(dot product) : 행렬 곱
# (x, 3) @ (3, y) >> (x, y)

print(f'{arr1.shape} @ {arr3.shape}')
print('내적 불가능')
# -

# arr1과 arr3의 내적을 위해 arr3의 형태 바꾸어 준다.
# T : 전치 행렬
print(f'{arr3.T}\n{arr3.T.shape}')

arr13 = arr1 @ arr3.T
print(f'{arr13}\n{arr13.shape}')

# ## numpy 배열에 함수 적용하기

a4 = np.array([0, 1, 2, 3])
print(f'array --> {a4}\nlist --> {[0, 1, 2, 3]}')

# sin 함수 적용
10 * np.sin(a4)

# numpy array method
third = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(third)
print(third.sum())
print(third.min())
print(third.max())

# 특정 행이나 열에 numpy method 적용하기
# 학생별 국어, 영어, 수학 점수 배열 활용
# 4명의 학생의 국영수 성적
stu_scores = np.array([[99, 93, 60],
                       [98, 82, 93],
                       [100, 56, 89],
                       [83, 49, 90]])
# 학생 4명의 과목별 총점
print(stu_scores.sum(axis=0))
# 각 학생의 과목 총점
print(stu_scores.sum(axis=1))
# 과목별 평균
print(stu_scores.mean(axis=0))
# 각 학생의 평균
print(stu_scores.mean(axis=1))

# ## 균일 분포에서 난수 생성
# * standard scaling : 데이터를 표준 정규 분포화하는 z-score 정규화
#     * 평균 = 0, 표준편차 = 1로 만들어주는 표준화 스케일링 기법
#     * 이상값에 민감하다.
#     * 분류 모델에 적합
#     * 수치형 변수에만 적용됨.
# * minmax scaler
#     * 변수의 범위를 바꿔주는 정규화 스케일링 기법 (0 ~ 1)
#     * 이상값에 민감하다.
#     * 회귀 모델에 적합하다.
#     * 이미지에서 쓰임
#     * 수치형 변수에만 적용

# seed 값 생성하지 않는 경우 --> 값을 출력할 때마다 바뀐다.
np.random.rand(5)

# * seed 값이 달라지는 것은 고정 값이 달라지는 것과 같은 의미이다.
# * 즉, 서로 다른 seed를 사용할 경우 numpy로부터 서로 다른 유사난수를 생성하게 한다.

# seed 값 생성 : 값이 고정됨.
np.random.seed(100)
np.random.rand(5)

# * rand() : 0 이상 1 미만 범위의 난수 array를 생성

np.random.rand(5, 3)

# 표준정규분포 normal distribution 난수 생성
# 평균 0, 표준편차 1인 정규분포를 바탕으로 생성
#  >> 음수 값이 나온다.
print(np.random.randn(5))
print(np.random.randn(5, 4))

# +
# 정규 분포 N(mu, signa^2)
mu, sigma = 0, 0.1
print(mu + sigma * np.random.randn(5))

# normal(평균, 표준편차, 데이터 개수)
print(np.random.normal(mu, sigma, 5))

# +
# unique()
nums = np.array([11, 12, 15, 17, 12, 26, 13])
print(nums)

# 중복 제거 & 오름차순 정렬
print(np.unique(nums))
print(len(np.unique(nums)))
# -

# 데이터프레임 만들기
data = {'city' : ['Naju', 'Yeosu', 'Jinju']}
df = pd.DataFrame(data)
print(df) # DataFrame
print(df['city']) # Series
print(df['city'].dtype)

data = {'country' : ['Korea', 'China', 'Japan', 'Germany', 'UK', 'USA']}
df = pd.DataFrame(data)
print(f'countries\n{df}')

data = {'country' : ['Korea', 'China', 'Japan', 'Germany', 'UK', 'USA', 'Japan', 'Germany', 'UK']}
df = pd.DataFrame(data)
print(df)
print(np.unique(df))

# ## 전치 행렬 : 행과 열을 바꿔준다.

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(arr)
print(arr.T)

# ## 다차원 배열의 평탄화

print(f'{arr} : {arr.ndim}D.')
print(f'{arr.flatten()} : {arr.flatten().ndim}D.')

# ## numpy 활용, csv 파일 읽기

# countries_data.csv 활용
path = '../data/countries_data.csv'
pd.read_csv(path, header=None)

# 컬럼명 넣기 1
raw = pd.read_csv(path, header=None,
                  names=['nation domain', 'nation', 'zip code', 'capital', 'code'])
df = raw.copy()
df

# +
# 컬럼명 넣기 2
df = pd.read_csv(path, header=None)
new_column = {0 : 'nation domain',
              1 : 'nation',
              2 : 'zip code',
              3 : 'capital',
              4 : 'code'}

df.rename(columns=new_column, inplace=True)
df

# +
# 1x2 그래프 그리기
# 그래프 크기
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

np.random.seed(100)
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

# violinplot 이용
# violinplot은 평균, 중앙값 둘 다 볼 수 있음. 선택해서 볼 수 있음.
axs[0].violinplot(all_data,
                  showmeans = False,
                  showmedians = True)
# 그래프 제목 설정
axs[0].set_title('Violin Plot')

# boxplot 이용
# boxplot은 원래 중앙값을 보여줌
axs[1].boxplot(all_data)
# 그래프 제목 설정
axs[1].set_title('Box Plot')

for ax in axs:
    # y축 그리드 설정
    ax.yaxis.grid(True)
    # x축의 눈금명 변경 (x축을 0으로 보고 시작 --> +1 을 해주어야 한다.)
    ax.set_xticks([y+1 for y in range(len(all_data))],
                  labels=['std6', 'std7', 'std8', 'std9'])
    # x축 이름 설정
    ax.set_xlabel('Four Separated Samples')
    # y축 이름 설정
    ax.set_ylabel('Observed Values')
    
# 그래프 보여주기
plt.show()
# -




