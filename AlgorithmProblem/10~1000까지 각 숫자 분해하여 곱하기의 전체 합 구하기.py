# http://codingdojang.com/scode/505?answer_mode=hide

# 숫자 각 digit 곱
def multi_digit(num):
    str_num = str(num)
    result = 1
    for chr in str_num:
        result = result * int(chr)
    return result

# a ~ b 범위에 숫자 각 digit 곱의 합
def sum_multi_digit_result(a, b):
    sum = 0
    for i in range(a, b + 1):
        sum += multi_digit(i)
    return sum

print(sum_multi_digit_result(10, 1000))

# 노가다 답
a = list(range(10, 1000))
sum = 0

for i in range(0, len(a)):
    if len(str(a[i])) ==2:
        sum += int(str(a[i])[0])*int(str(a[i])[1])
    elif len(str(a[i])) ==3:
        sum += int(str(a[i])[0])*int(str(a[i])[1])*int(str(a[i])[2])
    elif len(str(a[i])) ==4:
        sum +=int(str(a[i])[0])*int(str(a[i])[1])*int(str(a[i])[2])*int(str(a[i])[3])

print(sum)