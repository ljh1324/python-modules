for start, end in zip([1, 2, 3], [4, 5, 6]):
    print(str(start) + " "  + str(end))

for start, end in zip(range(0, 10, 2), range(10, 20, 2)):
    print(str(start) + " " + str(end))