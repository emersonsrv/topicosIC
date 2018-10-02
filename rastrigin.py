import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import differential_evolution


def rastrigin(x):
    return 10 * len(x) + sum([(y ** 2 - 10 * np.cos(2 * math.pi * y)) for y in x])


limit = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]

result = differential_evolution(rastrigin, limit)

print(rastrigin([0, 0, 0, 0]))
print(result.x)
print(result.fun)

X = np.linspace(-4, 4, 200)
Y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(X, Y)

Z = rastrigin([X, Y])

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
plt.show()
