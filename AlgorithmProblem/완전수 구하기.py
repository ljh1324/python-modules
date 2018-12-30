def is_complete_num(num):
    sum = 0
    for i in range(1, num):
        if num % i == 0:
            sum += i
    if sum == num:
        return True
    else:
        return False

def prt_all_complete_num(num):
    for i in range(1, num + 1):
        if (is_complete_num(i)):
            print(i)

num = int(input("input number: "))
prt_all_complete_num(num)
