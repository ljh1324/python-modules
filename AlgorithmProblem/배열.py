a = []
for i in range(5):
    b = []
    for j in range(i + 1):
        b.append(j + 1)
    a.append(b)

print(a)