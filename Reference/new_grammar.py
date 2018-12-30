flag = True
a = 0 if flag else 10
flag = False
b = 0 if flag else 10

print(a)
print(b)

arr = [num * 2 for num in range(10)]
print(arr)

arr = [num ** 2 for num in range(10)]
print(arr)

flag = 3 in arr
flag2 = 4 in arr
print(flag)
print(flag2)
