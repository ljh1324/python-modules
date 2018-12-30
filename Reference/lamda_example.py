sum = lambda a, b: a+b
print(sum(3,4))


myList = [lambda a,b:a+b, lambda a,b:a*b]
print(myList)
print(myList[0](3,4))
print(myList[1](3,4))
