# http://codingdojang.com/scode/406?answer_mode=hide

import math

def cal_need_page(num_of_post, post_per_page):
    return math.ceil(num_of_post / post_per_page)

print(cal_need_page(11, 10))