p = (1, 2, 3)
q = p[:1] + (5, ) + p[2:]
print(q)
print()

r = p[:1], 5, p[2:]     #make tuple using p
print(r)

p = (1, 2, 3)
q = (1, 2, 3)
print(p == q)
