# http://codingdojang.com/scode/484?answer_mode=hide

def is_upper(chr):
    return ('A' <= chr and chr <= 'Z')
def is_number(chr):
    return ('0' <= chr and chr <= '9')

def camel_case_to_pothole_case(camel_case):
    pothole_case = "";
    for i in range(len(camel_case)):
        if ( is_upper(camel_case[i]) ):
            pothole_case += ("_" + (camel_case[i].lower()))
        elif ( is_number(camel_case[i]) ):
            pothole_case += ("_" + (camel_case[i]))
        else:
            pothole_case += (camel_case[i])

    return pothole_case

print(camel_case_to_pothole_case("numGoat30"))


# 특이한 답1, 넣을 위치를 미리 찾아서 넣는 방식
import copy
def camtopot(string):
    stringlower = list(string.lower())
    string = list(string)
    stringresult = copy.deepcopy(stringlower)
    ilist = list()
    for i,x in enumerate(string):
        print(i, x)
        if x != stringlower[i]:     # 대문자일 경우 위치를 넣는다.
            ilist.append(i)
        elif 48 <= ord(x) <= 57:    # 숫자일 경우 위치를 넣는다.
            ilist.append(i)

    print(ilist)
    for i,x in enumerate(ilist):
        stringresult.insert(x,'_')  # stringresult의 길이가 1증가 함으로
        for j in range(i+1,len(ilist)): # 끼워 넣을 곳의 위치도 1씩 증가한다.
            ilist[j] += 1
    return stringresult

print(camtopot("numGoat30"))

# 특이한 답2.
def fpothole(S):
    c = -1
    for s in S:
        c = c + 1
        for i in range(65, 91):
            if ord(s) == i:
                S = S.replace(s, chr(i+32))
                S = S[:c] + "_" + S[c:]
                c += 1                          # "_" 를 추가했으므로 더해준다
        for i in range(48, 58):
            if ord(s) == i:
                S = S[:c] + "_" + S[c:]
                c += 1                          # "_" 를 추가했으므로 더해준다
    return S

print(fpothole("numGoatG30"))