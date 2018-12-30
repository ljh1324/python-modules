# 참고 Coding The Matrix - p.55

import plotting
import cmath

pi = cmath.pi
print(pi)
e = cmath.e
print(e)

n =  6

# e**(1j * theta) = cos(theta) + 1j*(sin(theta))    : 오일려 공식
w = [e**(2*pi*1j / (x + 1)) for x in range(n)]
print(w)

plotting.plot(w, 'ro', 4)