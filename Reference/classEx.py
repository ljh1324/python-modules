class Test:
    def __init__(self, name):
        self.name = name
    def print(self):
        print(self.name)

a = Test('A')
print(a.name)
Test.print(a)

b = Test('B')
print(b.name)
b.print()

c = Test('C')
print(c.name)



