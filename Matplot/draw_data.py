import matplotlib.pyplot as plt

f = open("powerful_jh_net.txt")

data = []

while True:
    input = f.readline()
    if not input: break;
    length = len(input)
    if input[length-1] == '\n':
        input = input[:length - 1]
    data.append(float(input))

data = data[:50]

idx = 0
max = 0
for i in range(len(data)):
    if max < data[i]:
        print('test')
        idx = i
        max = data[i]

length = len(data)
print(idx)
print(max)
plt.plot(data)
plt.axis([0, length, 0.5, 1.0])   # [xmin, xmax, ymin, ymax]
plt.text(idx, max, "max: " + str(max))
plt.ylabel("accuracy")
plt.show()



f = open("powerful_music_cnn.txt")

data = []

while True:
    input = f.readline()
    if not input: break;
    length = len(input)
    if input[length-1] == '\n':
        input = input[:length - 1]
    data.append(float(input))

data = data[:50]

idx = 0
max = 0
for i in range(len(data)):
    if max < data[i]:
        idx = i
        max = data[i]

length = len(data)
print(idx)
print(max)
plt.plot(data)
plt.axis([0, length, 0.5, 1.0])   # [xmin, xmax, ymin, ymax]
plt.text(idx, max, "max: " + str(max))
plt.ylabel("accuracy")
plt.show()

