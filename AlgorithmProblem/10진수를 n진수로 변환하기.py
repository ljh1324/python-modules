# http://codingdojang.com/scode/458?answer_mode=hide

def dec_to_n_radix(n, dec):
    if (n > 16 or n < 2):
        return 0
    numbers = "0123456789ABCDEF"
    n_radix = ""
    while True:
        n_radix = numbers[dec % n] + n_radix
        dec = int(dec / n)                      # int로 감싸주지 않았을 경우 0보다 항상 큰값이 되어 while문을 반복한다.
        if (dec <= 0):
            break
    return n_radix

print(dec_to_n_radix(16, 16))

# 깔끔한 답
def convert(n, base):
    T = "0123456789ABCDEF"
    q, r = divmod(n, base)
    if q == 0:
        return T[r]
    else:
        return convert(q, base) + T[r]

print(convert(16, 16))