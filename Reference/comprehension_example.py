a_set = {x * y for x in {1, 2, 3} for y in {2, 3, 4}}
print(a_set)

b_set = {x * y for x in {1, 2, 3} for y in {2, 3, 4} if x != y}
print(b_set)

a_list = [x * y for x in [1, 2, 3] for y in [10, 20, 30]]
print(a_list)

b_list = [[x, y] for x in ['A', 'B', 'C'] for y in [1, 2, 3]]
print(b_list)


a_list = [y for (x, y) in [(1, 'A'), (2, 'B'), (3, 'C')]]
print(a_list)

a_list = [x for x in range(10)]
print(a_list)

