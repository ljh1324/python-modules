dict_a = { k:v for (k,v) in [(3, 2), (4, 0), (100, 1)]}
dict_b = { (x, y): x*y for x in [1, 2, 3] for y in [1, 2, 3]}

print(dict_a)
print(dict_b)
