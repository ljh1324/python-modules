# http://codingdojang.com/scode/504?answer_mode=hide

def digit_count(start, end):
    count_arr = [0] * 10             # 0 ~ 9
    for i in range(start, end + 1):
        q = i
        while(1):
            q, r = divmod(q, 10)
            count_arr[r] += 1
            if q == 0:
                break;
    return count_arr

count_arr = digit_count(10, 15)
for i in range(10):
    print(str(i) + " " + str(count_arr[i]))

# 간단한 답
d = {}                    # dictionary 정의

for i in range(1,1001):

    for j in str(i):

        if d.get(j) == None:
            d[int(j)] = 1  # dic 의 key 값을 int로 하면 자동 정렬 됨
        else:
            d[int(j)] = d[j] + 1

print(d.items())
