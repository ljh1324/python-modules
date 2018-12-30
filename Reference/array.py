mylist = ["first element", 2, 3.14, 'forth element']
print(mylist[:])
print(mylist[0:2])
print(mylist[:-3])
print(mylist[:1])
print(mylist[-3:])
print(mylist[-4:3])
print(mylist[1:])
print(mylist[0])
print(mylist[0:1])
print(mylist[0:2:2])
print(mylist[0:4:2])

print()
mylist = ["first element", 2, 3.14]
mylist.append('forth element')
print(mylist)
#mylist.sort()       # string, int 형이 리스트안에 있기 때문에 정렬시 error!
#print(mylist)
print(mylist.index('first element'))
#print(mylist.index('not exist'))        # ValueError 발생!

mylist.pop(0)   # pop(index)
print(mylist)
mylist.remove('forth element')
print(mylist)
mylist.reverse()
print(mylist)
mylist.sort()       # int형태만 리스트안에 있기 때문에 정렬가능
print(mylist)
mylist.clear()
print(mylist)
mylist.append('b')
mylist.insert(1, 'a')
print(mylist)
mylist.sort()
print(mylist)