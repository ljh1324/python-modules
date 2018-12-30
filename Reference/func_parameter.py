def print_args(**kwlist):
    print(kwlist)

print_args(p1 = "1", p2 = "2", p3 = "3")
print_args(a = 1, b = 2, c = 3)

vals = {'p3':'3', 'p1':'1','p2':'2'}
print_args(**vals)

def add(*args):
    total = 0
    for i in args:
        total += i
    return total

vals = [1, 2]
print(add(*vals))

vals = (1, 2)
print(add(*vals))