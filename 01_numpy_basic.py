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

# 리스트와 넘파이의 차이
list_ = [1, 2, 3]
print(list_)

# +
# array : 배열
# np.array([]) 형태

arr = np.array([1, 2, 3])
print(arr) # 리스트와 같이 출력됨 (하지만 ,가 없음)
print(arr[0]) # 인덱싱 가능 = 순서가 있음.
print(arr[-1])
arr

# +
# 넘파이 배우는 이유 : 같은 위치의 원소끼리 연산이 가능하기 때문이다.

# 리스트 + 리스트 = 연장된 리스트
c = [1, 2, 3] + [4, 5, 6]
print(c)
print(len(c))

# +
arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])

# 같은 위치에 있는 값끼리 더한다. ==> 연산이 된다.
arr_a + arr_b
# -

# 여러 개의 리스트를 담으려면 []안에 여러 []를 작성해야 한다.
arr_e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 2차원 --> 행렬
print(arr_e[-1])
print(arr_e[-1][-1])

# ## Numpy 배열 속성

# shape : 배열의 형태 확인하기 (행, 열)
print(arr_e)
arr_e.shape

print(arr_a)
arr_a.shape # 1차원

# 질문
arr_f = np.array([[1], [2], [3]])
print(arr_f)
arr_f.shape # 2차원

arr_m = np.array([[1, 2, 3], [4, 5, 6]])
arr_m.shape

# ndim(number of dimension) : 배열의 차원 수
print(arr_f.ndim)
print(arr_a.ndim)

# dtype(data type) : 요소의 자료형
arr_f.dtype

# itemsize : 요소의 한 개의 크기
arr_f.itemsize

# size = 전체 요소의 개수
arr_f.size

np.zeros((3, 4))

print(np.ones((3, 4)))
print(np.ones((3, 4)).dtype)

# 항등행렬
np.eye(3)

# +
# arange() (array + range) : 배열의 범위
# np.arange(start, stop, step)

np.arange(5)
# -

print(np.arange(5))

np.arange(1, 6)

np.arange(1, 10, 2)

# linspace()
# 0부터 10까지에서 동등하게 10개로 균등하게 나눠주기
np.linspace(0, 10, 10)

np.linspace(1, 10, 5)

# +
# 2개의 배열 합치기
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print(x)
print(y)
# -

# concatenate() : 배열 합치기 (default : axis=0)
concat_0 = np.concatenate((x, y))
concat_1 = np.concatenate((x, y), axis=1)
print(concat_0)
print(concat_0.shape)
print(concat_1)
print(concat_1.shape)

# +
# vstack, hstack
# vstack(vertical stack) = concatenate(axis=0)
# hstack(horizontal stack) = concatenate(axis=1)

print(np.vstack((x, y)))
print(np.hstack((x, y)))
# -

# ## reshape
# * 차원 변화, 형태 변화

a = np.arange(12)
print(a)
print(a.shape) # 1차원

# reshape() : 형태 변경
# 1차원 >> 2차원
print(a)
print(a.reshape(3, 4))

dim_2 = a.reshape(3, 4)
dim_2.shape # 2차원

# 형태 변경할건데, 행은 3개, 열은 전체 원소의 개수를 고려해서 자동으로 맞춰주기
# 열의 개수를 정해주고 행을 맞춰줄수도 있음.
a.reshape(3, -1)

temp = np.arange(120)
temp

temp.reshape(3, -1)

# 나누어떨어지지 않으면 error
temp_1 = np.arange(11)
temp_1

temp_1.reshape(2, -1)

# +
# 배열 분할하기

arr_30 = np.arange(30).reshape(-1, 10)
print(arr_30)
# -

# 컬럼의 인덱스가 3을 기준으로 분할하기
# 분할해서 따로 저장할 수 있음.
arr_30_01, arr_30_02 = np.split(arr_30, [3], axis=1)
print(arr_30_01)
print(arr_30_02)

# +
# newaxis : 배열에 새로운 축 추가하기
a = np.array([1, 2, 3, 4, 5, 6])

print(a)
print(a.shape) # 1차원
# -

a1 = a[np.newaxis, :]
print(a1)
print(a1.shape) # 1x6 행렬 (2차원)

a2 = a[:, np.newaxis]
print(a2)
print(a2.shape) # 6x1 행렬 (2차원)

# +
# 인덱스와 슬라이싱
ages = np.array([18, 19, 25, 30, 28])
print(ages)

# 인덱싱
print(ages[2])

# 슬라이싱
print(ages[1:3])
print(ages[:-2])

# +
# 논리적 인덱싱
condition = (ages > 20) # 비교 연산자

# 조건이 True인 것만 뽑아줘 !
ages[condition]
# -

# 2차원 배열 인덱싱
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
print(a[0])
print(a[0][2])
print(a[0, 2])

# 값 수정 가능
a[0, 0] = 12
print(a)

print(a[0:2])
print(a[0:2].shape)

print(a)
print(a[0:2, 1:3])

# ### 실무코드

a[::2, ::2]

# ## Numpy 연산

arr1 = np.array([[1, 2], [3, 4], [5, 6]])
arr2 = np.array([[1, 1], [1, 1], [1, 1]])
print(arr1)
print(arr2)

result = arr1 + arr2
print(result)

# list + list
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
print(a + b)

a.extend(b)

print(a)

# +
# 브로드 캐스팅

np.array([1, 2, 3]) # (3, ) 1차원
# -

miles = np.array([1, 2, 3])
print(miles)
print(miles * 1.6)
print(miles[0] * 10)

# 넘파이 곱셈
arr1 = np.array([[1, 2], [3, 4], [5, 6]])
arr2 = np.array([[2, 2], [2, 2], [2, 2]])
result  = arr1 * arr2
print(arr1)
print(arr2)
print(result)

# +
# 내적 (dot product) : 행렬 곱
# 입력 행렬 열 @ 출력되는 행이 같아야 한다.
# ex) (x, 3) @ (3, y) >> (x, y)

print(arr1.shape)
print(arr2.shape)

# +
## ValueError: matmul 행렬 내적을 위한 행과 열의 크기가 맞아야 계산 가능함.
# arr1 @ arr2
# -

# arr1와 arr2의 내적을 위해 arr2의 형태를 바꾸어 준다.
# T (transpose) 전치 행렬
print(arr2.T)
print(arr2.T.shape)

arr3 = arr1 @ arr2.T
print(arr3)
print(arr3.shape)

# +
# numpy 배열에 함수 적용하기

a = np.array([0, 1, 2, 3])
print(a)
print([0, 1, 2, 3])
# -

# sin 함수에 적용하기 >> array의 요소에 각각 접근 (Broad casting)
10 * np.sin(a)

# +
# numpy array method

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)

# array의 원소끼리 더해줌.
print(a.sum())

# array의 원소 중 최솟값/최댓값 구하기
print(a.min())
print(a.max())

# +
# 특정 행이나 열에 numpy method 적용하기
# 학생 별 국어, 영어, 수학 점수 배열 활용

# 4명의 학생의 국영수 성적
stu_scores = np.array([[99, 93, 60], [98, 82, 93], [93, 65, 81], [78, 82, 81]])

# 4명의 학생의 과목별 총점 구하기
print(stu_scores.sum(axis=0))

# 학생별 총점 구하기
print(stu_scores.sum(axis=1))

# 과목별 평균 구하기
print(stu_scores.mean(axis=0))

# 학생별 평균 구하기
print(stu_scores.mean(axis=1))
# -

# ## 균일 분포에서 난수 생성
# * 과학적 통계방법 : random
#     * rand : 무작위하게 0 ~ 1 사이의 값에서 정한 수만큼의 값을 추출
# * 정규분포
#     * m(=mu) : 평균
#     * sigma : 편차 (평균에서 떨어진 정도)

# seed 값 생성 안 하는 경우 : 값을 출력할 때마다 바뀐다.
np.random.rand(5)

# seed 값 생성 : 값이 고정된다.
np.random.seed(100)
np.random.rand(5)

# (행, 열)
np.random.rand(5, 3)

# 표준정규분포 (normal distribution) 난수 생성
# 평균 0, 표준편차 1인 정규분포를 바탕으로 생성
# >> 음수 값이 나온다.
np.random.randn(5)

# (행, 열)
np.random.randn(5, 4)

# +
# 표준정규분포

mu, sigma = 0, 0.1
print(mu + sigma * np.random.randn(5))
print(np.random.normal(mu, sigma, 5))

# +
# 고유항목 및 개수를 얻는 방법
a = np.array([11, 12, 23, 56, 32, 15, 11, 72, 14])
print(a)

# 중복제거 + 오름차순 정렬
print(np.unique(a))
print(len(np.unique(a)))
# -

import pandas as pd
import numpy as np

# +
# 데이터프레임 만들기
data = {'name' : ['Naju', 'Yeosu', 'Jinju']}
df = pd.DataFrame(data)

print(df) # DataFrame
print(df['name']) # Series
print(df['name'].dtype)

# +
data = {'name' : ['Ulsan', 'Daegu', 'Busan', 'Sangju', 'Daegu', 'Busan', 'Sangju']}
df = pd.DataFrame(data)

print(df)
print(np.unique(df))
# -

# ## 전치 행렬 (행과 열을 바꾸어줌)

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(arr)
print(arr.T)

# ## 다차원 배열의 평탄화

print(arr)
print(arr.flatten())

print(arr.shape) # 2차원
print(arr.flatten().shape) # 1차원

# ## Numpy 활용, csv 파일 읽기

# countries_data.csv 활용
path = '../datasets/countries_data.csv'
pd.read_csv(path)

pd.read_csv(path, header=0)

# names = [] : 컬럼명 넣기
raw1 = pd.read_csv(path, header=None,
                  names=['nation_domain', 'Nation', 'zip_code', 'capital', 'code']) # header=None : 첫 번째 행부터 data로 인식하고 싶을 때 사용
# cf. header=0 : index의 0을 컬럼의 이름으로 사용함.
df1 = raw1.copy()
df1.head()

df = pd.read_csv(path, header=None)
df

# +
# 컬럼명 넣기 : dict 이용하는 방법
new_columns = {0 : 'nation_domain', 1 : 'Nation', 2 : 'zip_code', 3 : 'capital', 4 : 'code'}

df.rename(columns=new_columns, inplace=True)
df
# -

path = '../datasets/countries_header.csv'
pd.read_csv(path)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# * grid : 격자, tick : 눈금자, label : 축 이름, legend : 범례

# +
# 1행 2열으로 그래프 그릴 것
# 그래프 사이즈 : figsize
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

np.random.seed(100)
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

# violinplot 이용
# violinplot은 평균, 중양값 둘 다 볼 수 있다. 선택해서 볼 수 있음.
axs[0].violinplot(all_data,
                  showmeans = False, showmedians = True)
# 그래프 제목 설정
axs[0].set_title('Violin plot')

# boxplot 이용
# boxplot은 원래 중앙값을 보여준다.
axs[1].boxplot(all_data)
# 그래프 제목 설정
axs[1].set_title('Box plot')

for ax in axs:
    # y축 그리드 설정
    ax.yaxis.grid(True)
    # x축 눈금 이름 변경 (x축 = 0)
    ax.set_xticks([y+1 for y in range(len(all_data))],
                  labels=['std6', 'std7', 'std8', 'std9'])
    # x축 이름 설정
    ax.set_xlabel('Four separated samples')
    # y축 이름 설정
    ax.set_ylabel('observed values')
    
# 그래프 보여주기
plt.show()
# -

print(len(all_data))
all_data

# ## Q&A

rng = pd.date_range("4/23/2024 00:00", periods = 5, freq = "D")
print(rng)
rng + pd.offsets.BusinessDay(5)

# +
rng = pd.date_range("4/23/2024 00:00", periods = 5, freq = "D")

# rng 출력
print(rng)

# BusinessDay offset 적용
rng_biz_day = rng + pd.offsets.BusinessDay(5)
print(rng_biz_day)


# +
def apply_biz_day(dates, offset):
    result_dates = []
    for date in dates:
        # 주말 : 날짜 추가 없음 >> 다음 날짜 계산
        if date.weekday() >= 5: # 5 : 주말
            modified_date = date + pd.offsets.BusinessDay()
        else:
            modified_date = date
    
        result_dates.append(modified_date + pd.offsets.BusinessDay(offset))
            
    return pd.DatetimeIndex(result_dates)

# 5개의 BusinessDay 적용
apply_biz_day(rng, 5)
# -


