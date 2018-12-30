# http://codingdojang.com/scode/405?answer_mode=hide

def make_str(chr, num_of_chr):
    result = ''.join([' ' for i in range(num_of_chr)])
    #for i in range(num_of_chr):
    #    result += chr
    return result

def tab_to_space(code, num_of_space):
    str_space = make_str(' ', num_of_space)
    code = code.replace('\t', str_space)
    return code

code = """for i in range(100)
\tprint(i)
\tprint(i)"""

print(code)
reformat_code = tab_to_space(code, 4)
print(reformat_code)

# 디테일한 답
filename=input("Enter your file name : ")
tempfile=open(filename)
tempfile=tempfile.read()
temp_str=tempfile.replace("\t","    ")
tempfile=open(filename,'w')
tempfile.write(temp_str)
tempfile.close()