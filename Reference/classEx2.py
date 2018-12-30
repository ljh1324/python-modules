class Adder:
    num1 = 0
    num2 = 0
    __private = 10
    def __init__(self, a = 0, b = 0):
        self.num1 = a
        self.num2 = b

    def create(a = 0, b = 0):
        person = Adder(a, b)
        return person

    def set(self, a, b):
        self.num1 = a
        self.num2 = b
    def result(self):
        return (self.num1 + self.num2)
    def __str__(self):
        return(str(self.num1) + " " + str(self.num2))

a = Adder()
print(a.num1, a.num2)
print(a)
#print(a.__private)

b = Adder(1, 2)
print(b.num1, b.num2)
print(b)

c = Adder.create(10, 10)
print(c)

r = a.result()
r2 = Adder.result(b)

print("A Result: " + str(r))
print("B Result: " + str(r2))

