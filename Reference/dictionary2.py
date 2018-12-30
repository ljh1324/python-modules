a = {'name' : 'lee', 'age' : 23, 1:'A', (1, 2):'tuple!'}

print(a.keys())
print(a.values())
print(a.items())

print()

print(list(a.keys()))
print(list(a.values()))
print(list(a.items()))

try:
    print(a['nothing'])
except KeyError as e:
    print("Not Present!")

flag = 'Present!' if 'nothing' in a else 'Not Present!'
flag2 = 'Present!' if 'name' in a else 'Not Present!'
print(flag)
print(flag2)


