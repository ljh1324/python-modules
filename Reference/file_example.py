f = open("file.txt", 'w')
for i in range(1, 11):
    data = u"%d번째 줄입니다.\n" % i      # u: 유니코드로 출력
    f.write(data)
f.close()


f = open('file.txt', 'r')
lines = f.readlines()

print(lines)
for i in range(len(lines)):
    lines[i] = lines[i][:lines[i].rfind('\n')]
print(lines)

f = open('file.txt', 'r')
lines = f.read()
print(lines)