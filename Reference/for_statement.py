langs = ['python', 'ruby', 'nodejs']
for l in langs:
    print(l, len(l))

for i in range(1, 10):
    for j in range(1, 10):
        print("{} x {} = {}".format(i, j, i * j))

a = {1 : 'a', 2 : 'b'}
for i in a:
    print(a)

# a의 key, val을 동시에 조회한다
for k, v in a.items():
    print(k, v)

# a의 key 목록만 조회한다
for k in a.keys():
    print(k)

# a의 값만 조회한다
for v in a.values():
    print(v)
