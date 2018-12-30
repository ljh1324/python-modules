# http://codingdojang.com/scode/414?answer_mode=hide

def special_sorting(arr):
    minus_list = []
    plus_list = []
    for i in arr:
        if (i >= 0):
            plus_list.append(i)
        else:
            minus_list.append(i)
    result_list = minus_list + plus_list
    return result_list

print(special_sorting([-1, 1, 3, -2, 2]))
