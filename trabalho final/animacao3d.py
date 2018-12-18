import pandas
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
import sys

edges = [[20, 1, 2, 1,  8, 10, 2,  9, 11, 3, 4, 7, 7,  5, 6,  14, 15, 16, 17],
		[  3, 3, 3, 8, 10, 12, 9, 11, 13, 4, 7, 5, 6, 14, 15, 16, 17, 18, 19]]

basePath = '../../MSRAction3D/skeleton/'

def loadData(action, subject, instance):
	ins = np.loadtxt(basePath + 'a%02i_s%02i_e%02i_skeleton.txt'%(action, subject, instance))
	ins = ins.reshape((ins.shape[0]//20, 20, 4))
	return ins

if len(sys.argv) >= 4:
	action = int(sys.argv[1])
	subject = int(sys.argv[2])
	instance = int(sys.argv[3])
else:
	action = 1
	subject = 1
	instance = 1

data = loadData(action, subject, instance)
print(data.shape)

fig = plt.figure()
ax = p3.Axes3D(fig)

# Setting the axes properties
ax.set_xlim3d([0.0, 300.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1400.0])
ax.set_ylabel('Z')

ax.set_zlim3d([300.0, 0.0])
ax.set_zlabel('Y')
#graph = ax.scatter(data[1,:,0], data[1,:,2], data[1,:,1])

graph = ax.scatter([], [], [])


def animate(i):
	graph._offsets3d = (data[i,:,0], data[i,:,2], data[i,:,1])
	return graph

ani = FuncAnimation(fig, animate, frames=data.shape[0], interval=100)
plt.show()


