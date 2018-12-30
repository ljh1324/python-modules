import numpy as np

# arange() : 1차원 배열을 선언한다. reshape() : n차원 배열의 모양을 변경한다.
print()
print('arange(), reshape()')
arr = np.arange(15)
print(arr)
arr = arr.reshape(3, 5)
print(arr)

# shape: 배열에서 각 차원의 크기를 알려준다. dtype: 배열에 저장된 자료형을 알려준다.
print()
print('shape, dtype')
print(arr.shape)
print(arr.dtype)

# 리스트를 생성한 후 리스트를 인자로 넘겨서 1차원 ndarray 생성 후 reshape()
print()
print('리스트를 인자로 넘겨서 1차원 ndarray 생성 후 reshape()')
data = [[1,2,3,4],[5,6,7,8]]
arr = np.array(data)
print(arr)

# 초기화되거나 되지않은 ndarray 생성
print()
print('초기화되거나 되지않은 ndarray 생성')
arr2 = np.zeros(16).reshape(4,4)
arr3 = np.ones(16).reshape(4,4)
arr4 = np.empty(16).reshape(4,4)
print('zeros() : \n', arr2)
print('ones() : \n' , arr3)
print('empty() : \n', arr4)

arr5 = np.ones((4,4))
arr6 = np.empty((2,3,3))
arr7 = np.eye(4)
print('ones((4,4)) : \n', arr5)
print('empty((2,3,3)) : \n', arr6)
print('eye(4) : \n' , arr7)

#numpy의 데이터 타입은 C나 포트란과 같은 저수준 언어로 작성된 코드와 연동될 수 있다.
## ndarray를 생성할 때 매개변수로 'dtype'을 줄 수 있다. 데이터 타입의 형식은 다양하다. numpy 모듈에 상수로 정의되어 있다.
print()
print('ndarray를 생성할 때 매개변수로 "dtype"을 줄 수 있다.')

ar= [[2,3,4,5],[3,4.3,2.1,4]]
arr = np.array(ar)
print(arr.dtype)
ar= [[4,5,6,3],[0,0,0,0]]
arr = np.array(ar)
print(arr.dtype)

arr = np.array(ar,dtype=np.float64)
print('modify dtype :',arr.dtype,'result :\n',arr)

arr = np.array(ar,dtype=np.string_)
print('modify dtype :', arr.dtype,'result :\n',arr)

# 데이터형 강제 캐스팅이 가능한대 ndarray 함수인 astype()에 넣어주면된다.
print()
print('데이터형 강제 캐스팅이 가능한대 ndarray 함수인 astype()에 넣어주면된다.')
arr.astype(np.unicode_)
print(arr)

# 배열의 스칼라 연산
print()
print('배열의 스칼라 연산')
arr = np.array([[1,2,3,4],[10,20,30,40]],dtype=np.float32)
print(arr + arr)
print(arr - arr)
print(arr*2)
print(arr/arr)
print(arr/2)
print(arr**2)

# 색인과 슬라이싱 기초
# 1차원 배열은 파이썬의 리스트와 비슷하게 동작한다. 다만 더많은 특성과 차이점을 가지고 있다.
# 1)배열에 슬라이싱을 할경우 새로운 객체가 생기는것이 아닌 기존의 객체의 뷰가 생성된다.
#  2)배열의 슬라이싱에 직접 데이터를 할당할 수 있다. 이를 브로드 캐스팅이라 한다. 이는 대용량 데이터처리를 염두해두고 설계했기 때문에 메모리 복사를 미리 방지한다.
print()
print('색인과 슬라이싱 기초')
arr=np.ones(16)
print(arr)
arr[0:4] = 10
print(arr)
arr1 = arr[0:4]
arr1[:]=30
print(arr)

""" 출력 결과
[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
[ 10.  10.  10.  10.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
   1.]
[ 30.  30.  30.  30.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
   1.]
"""

# 다차원 배열을 다루기 위해선 더 많은 인덱싱 방법이 필요하다.
print()
print('다차원 배열을 다루기 위해선 더 많은 인덱싱 방법이 필요하다.')
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[0:2])
arr2d[0][0] = 99
arr2d[1][0:2] = 44
print(arr2d)

# 일반 배열도 narray처럼 인덱싱 가능하다.
ar=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(ar[0:2])

# 깊은복사
print()
print('깊은 복사')
copy_ = arr2d[:].copy()
print(copy_)

# 3차원 배열의 색인
print()
print('3차원 배열의 색인')
arr = np.zeros((2,4,4))
print(arr)
print('arr.shape :',arr.shape)
arr[0][0][0:4] = 3
print(arr[0])
arr[1]=-1
print(arr[1])

# 2차원 배열에서 슬라이싱을 2번 사용하면 같은 차원의 배열을 얻게된다.
# 슬라이싱을 1번 사용하면 한차원 낮은 배열을 얻게된다. 슬라이싱은 0번 사용하면 2차원 낮은 배열을 얻게된다.
print()
print('2차원 배열 슬라이싱')
arr = np.arange(16).reshape((4,4))
print(arr.dtype)
print(arr)
print(arr[:2,:1])
print(arr[:2,3])
print(arr[0,:1])
print(arr[0,3])

# 불리언 색인
print()
print('불리언 색인')
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(names)
print(names=='Bob')

arr = np.array([names,names])
barr = arr=='Joe'
print(barr)

data = np.arange(28).reshape((7, 4))
print(data[names == 'Bob'])             # Bob이 있는 열인 0, 3 반환후 data의 0행 3행 출력
print(data[(names!='Bob') & (names!='Joe')])

# 팬시 색인
# 팬시 색인이란 정수 배열을 사용한 색인을 뜻한다.
# 찾기 원하는 행들의 번호를 리스트에 담아서 넘겨서 색인을 할 수 있다.
# 다른 리스트를 한개더 넘길 경우 각 행에서 각 열의 값을 뽑아온다.
# 뽑아온 행들에서 [:, 리스트] 형태로 넘길경우 리스트에 들어있는 색인값 순서대로 열의 순서가 바뀌어 출력된다.
print()
print('팬시 색인: 정수 배열을 사용한 색인')
arr = np.arange(40).reshape((8,5))
print(arr)
print(arr[[1, 2, 3, 4]])
print(arr[[1, 2, 3, 4]][[0, 2]])                # 1, 2, 3, 4행 에서 0행~1행 출력
print(arr[[1, 2, 3, 4],[3, 3, 4, 4]])
print(arr[[1, 2, 3, 4]][:, [4, 1, 2, 3, 0]])

# 배열 전치
print()
print('배열 전치')
arr = np.arange(40).reshape(8,5)
print(arr)
print(arr.T)
print(np.dot(arr.T, arr))

# 배열 곱하기
print()
print('배열 곱하기')
arr = np.eye(4)         # 4 by 4 단위행렬
print(arr)
arr2 = np.arange(16).reshape((4, 4))
print(arr2)
print(np.dot(arr, arr2))

# 유니버셜 함수
print()
print('유니버셜 함수')
arr = [x**2 for x in range(1, 17)]
print(arr)
arr2d = np.array(arr).reshape((4, 4))
print(arr2d)
print(np.sqrt(arr2d))
print(np.exp(np.sqrt(arr2d)))

arr = np.zeros((4, 4))
print(arr)
arr2 = np.zeros((4, 4)) + 1
print(arr2)
print(np.maximum(arr, arr2))

arr = np.random.randn(25).reshape((5, 5))
print(arr)
arr = np.random.randn(5, 5)
print(arr)

arr = np.arange(10)
print(arr)
print(np.modf(arr))




