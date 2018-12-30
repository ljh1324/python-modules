import matplotlib.pyplot as plt
import numpy as np

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()


plt.plot([1,2,3,4], [1,4,9,16], 'ro-')
plt.axis([0, 6, 0, 20])   # [xmin, xmax, ymin, ymax]
plt.show()

t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
# plot(x, y, 'format1', x2, y2, 'format2', ,,,)  # plot x and y using blue circle markers
plt.plot(t, t, 'r--', t, t**2, 'bs-', t, t**3, 'g^')
plt.show()

plt.plot([1, 2, 3, 4], [1, 2, 3, 4], 'ro-', linewidth=5.0)
plt.show()

plt.hist([10, 20, 30, 30, 30, 40, 50], bins=100, color='b')
plt.show()
