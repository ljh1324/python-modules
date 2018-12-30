class myClass1:
    def __init__(self, num=None):
        self.num = 0 if num is None else num
    def __str__(self):
        return 'myClass1의 인스턴스 입니다'

test = myClass1()
print(test.num)
test2 = myClass1(10)
print(test2.num)
print(test2)

print()

class myClass2:
    def function(self, *arglist):
        print(arglist)
        if not arglist:
            print('no arguments')
        else:
            for arg in arglist:
                if type(arg) == str:
                    print("string!")
                elif type(arg) == list:
                    print("list!")
                elif type(arg) == int:
                    print("int!")

test = myClass2()
arr = [1, 2, 3]
test.function(arr, 1, 'a')
test.function()


