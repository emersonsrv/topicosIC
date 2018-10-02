import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def rosen(parametros):
    x1, x2 = parametros
    return 100.0 * (x1 - x2 ** 2) ** 2.0 + (x2 - 1) ** 2


a = np.arange(-2, 3)
b = np.arange(-2, 3)
y = rosen([a, b])
print(a)
print(b)
print(y)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(a, b, y)
plt.show()
