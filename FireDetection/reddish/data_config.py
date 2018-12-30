file = open('reddish_compose_image.txt', 'r')

data = [int(x) for x in file.read().split(' ')]

numImage = data[0]
itemSize = data[1] * data[2] * data[3]
splitN = 2
length = len(data)
print(len(data[4:]))

idx = 0
check = 1
for i in range(splitN):
    file2 = open('reddish_compose_image_' + str(i) + '.txt', 'w')
    file2.write(str(numImage / splitN) + ' ')
    file2.write(str(data[1]) + ' ')
    file2.write(str(data[2]))

    start = int(numImage * itemSize / splitN * i)
    j_size = int(numImage * itemSize / splitN)

    print(start)
    print(j_size)
    for j in range(j_size):
        file2.write(' ' + str(data[start + 4 + j]))
    file2.close()

