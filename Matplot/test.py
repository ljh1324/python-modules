import matplotlib.pyplot as plt
arr = [0.7727272727272727, 0.7058823529411765, 0.7132352941176471, 0.6985294117647058, 0.7132352941176471, 0.8974358974358975, 1.0, 1.0, 1.0]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.bar(x, arr)
plt.ylabel('accuracy')
plt.xlabel('models')
plt.xticks(x, x)
#plt.ylim([0, 1])
plt.show()

