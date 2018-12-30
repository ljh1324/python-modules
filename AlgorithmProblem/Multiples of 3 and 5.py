# http://codingdojang.com/scode/350#answer-filter-area

def get_gcd(u, v):                    # 최대공약수
    if v == 0:
        return u
    return get_gcd(v, u % v)

def get_lcm(u, v):                    # 최소공배수
    lcm = int(u * v / get_gcd(u, v))
    return lcm

def get_sequence_sum(n, d):         # 등차 수열의 합
    last = int(n / d) * d
    q = int(n / d)
    sequence_sum = q * (last + d) / 2
    return int(sequence_sum)

a = 3
b = 5
n = 1000
lcm = get_lcm(a, b)
print(get_sequence_sum(n - 1, a) + get_sequence_sum(n - 1, b) - get_sequence_sum(n - 1, lcm))

# 기발한 답
print(sum(list([x for x in range(1000) if x%3==0 or x%5==0])))

# 기발한 답2
set3 = set(range(3, 1000, 3))
set5 = set(range(5, 1000, 5))

print(sum(set3 | set5))