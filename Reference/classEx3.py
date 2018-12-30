class TestClass:
    def __init__(self, num1, num2):
        self.__var = num1
        self.var2 = num2
    def getNum1(self):
        return self.__var

    def create(num1, num2):
        testClass = TestClass(num1, num2)
        return testClass

test = TestClass.create(10, 10)
print(test.__var)              # error
print(test.var2)
print(test.getNum1())