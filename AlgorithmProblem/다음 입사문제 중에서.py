# http://codingdojang.com/scode/408?answer_mode=hide

import numpy as np

def get_min_distance_dot(arr):
    arr = np.array(arr)

    length = len(arr)
    arr1 = arr[:length - 1]
    arr2 = arr[1:]

    result = arr2 - arr1

    min_distance_dots = []
    min_val = min(result)

    min_distance_dots = [(arr1[i], arr2[i]) for i in range(len(result)) if result[i] == min_val]

    #for i in range(len(result)):
    #    if (min_val == result[i]):
    #        min_distance_dots.append((arr1[i], arr2[i]))
    return min_distance_dots

dots = get_min_distance_dot([1, 3, 4, 5, 13, 17])
for a, b in dots:
    print("(", a, ",", b, ")")


# 깔끔한 답

#데이터
s = [1,3,4,8,13,17,20]

#처리
def short(src):
    distance = []
    for k in range(len(src)-1):
        distance.append(src[k+1]-src[k])
    m = min(distance)
    for l in range(len(distance)):
        if distance[l] == m :
            return print(src[l],src[l+1])

#입출력
short(s)


# 복잡한 답
data = [1, 3, 4, 8, 13, 17, 20, 22, 23, 24, 25,31,32,35,76,546,547]
shortest_list=[]

def shortest(list):
    n = 1
    result_list = []
    for i in range(len(list)-1):
        result_list.append(list[i+1] - list[i])             # 거리를 result_list에 저장
    index = result_list.index(min(result_list))             # 최솟값의 index를 찾는다
    kk=result_list[index]                                   # 최솟값 저장
    shortest_list.append([list[index], list[index+1]])      # 최솟값의 item을 shortest_list에 저장
    del result_list[index]                                 # 최솟값 삭제

    new_index = result_list.index(min(result_list))         # 새로운 최솟값의 index를 찾는다
    nkk=result_list[new_index]                              # 새로운 최솟값 저장

    while kk==nkk:                                     # 최솟값과 같을 경우
        shortest_list.append([list[new_index+n], list[new_index +(n+1)]])   # 붙여준다.
        del result_list[new_index]                     # 붙여주고 새로운 최솟값 삭제. kk=nkk를 언젠가는 끝내야 하기 때문에 값을 삭제해주는게 필요하다
        new_index = result_list.index(min(result_list))
        nkk=result_list[new_index]
        n=n+1                                           # n을 더해주는 이유: result_list에서 하나를 삭제했으므로?
    return(shortest_list)

print(shortest(data))