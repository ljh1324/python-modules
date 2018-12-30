a = {'name' : 'lee', 'age' : 23, 1:'A', (1, 2):'tuple!'} # list는 key로 들어갈 수없음!
print(a['name'])
print(a['age'])
print(a[1])
print(a[(1, 2)])

print()
a[2] = 'B'
print(a[2])

print()
for key in a:
    print(key)

print()
del a['name']
for key in a:
    print(key)