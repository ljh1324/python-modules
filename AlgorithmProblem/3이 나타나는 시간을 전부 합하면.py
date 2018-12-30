# http://codingdojang.com/scode/473?answer_mode=hide

def comprise_three(num):
    while(1):
        num, r = divmod(num, 10)
        if (r == 3):
            return True
        if (num == 0):
            break
    return False

def cal_seconds():
    comprise_three_seconds = 0          # 0~59에서 3이 포함된 총 시간(초)
    for i in range(60):
        if (comprise_three(i)):
            comprise_three_seconds += 60
    return comprise_three_seconds

def cal_total_seconds_comprise_three():
    seconds_per_hour = 3600
    comprise_three_seconds = cal_seconds()
    total_seconds_comprise_three = 0
    for i in range(24):
        if (comprise_three(i)):
            total_seconds_comprise_three += seconds_per_hour         # Ex. 3시 00분 ~ 3시 59분, 3이 시간에 포함됨!
        else:
            total_seconds_comprise_three += comprise_three_seconds   # Ex. 0시 00분 ~ 0시 59분. 3이 시간에 포함이 안될 수 있음!

    return total_seconds_comprise_three

print(cal_total_seconds_comprise_three())

# 간단한 답
def clock(num):
    sec = 0
    for h in range(24):
        for m in range(60):
            if str(num) in str(h) + str(m): # 문자열로 변환한 시간에 해당 숫자가 들어있으면
                sec += 60
    return sec

print(clock(3))