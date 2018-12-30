S = [2 + 2j, 3 + 2j, 1.75 + 1j, 2 + 1j, 2.25 + 1j, 2.5 + 1j, 2.75 + 1j, 3 + 1j, 3.25 + 1j]

import cmath

from plotting import plot

plot(S, 'ro--', 4)

S2 = [-5 + 2j +z for z in S]     # 평행이동

plot(S2, 'ro--', 4)

S3 = [z / 2 for z in S2]

plot(S3, 'ro--', 4)

# (x, y)를 90도 회전
# z * j = (x + yj) * j = (xi * y * j * j) = (xi - y) = (-y, x)
S4 = [z * 1j for z in S3]

plot(S4, 'ro--', 4)


# 참고 Coding The Matrix - p.56
delta = cmath.pi / 4
print(cmath.e ** (delta * 1j))
S5 = [z * (cmath.e ** (delta * 1j)) for z in S]

plot(S5, 'ro--', 4)

