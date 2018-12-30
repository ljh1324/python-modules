data = [1, [2, 3, 4], 5]

# 얕은 복사
s_copied = data.copy()

# 깊은 복사
import copy
d_copied = copy.deepcopy(data)

#원본 데이터 수정
data[1][1] = 'xx'
print(s_copied)
print(d_copied)

