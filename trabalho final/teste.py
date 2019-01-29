import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time, math
import scipy.stats as stats

edges = [[20, 1, 2, 1,  8, 10, 2,  9, 11, 3, 4, 7, 7,  5, 6,  14, 15, 16, 17],
		[  3, 3, 3, 8, 10, 12, 9, 11, 13, 4, 7, 5, 6, 14, 15, 16, 17, 18, 19]]

basePath = '../../MSRAction3D/skeleton/'

def loadData(action, subject, instance):
	ins = np.loadtxt(basePath + 'a%02i_s%02i_e%02i_skeleton.txt'%(action, subject, instance))
	ins = ins.reshape((ins.shape[0]//20, 20, 4))
	return ins

#Cria dicionario com toda a base de dados
def loadBase():
	base = {}
	for action in range(1, 21):
		base[action] = {}
		for subject in range(1, 11):
			base[action][subject] = {}
			for instance in [1, 2, 3]:
				try:
					data = loadData(action, subject, instance)
				except IOError:
					continue
				base[action][subject][instance] = data

	return base

def framesByCoords(action, coords):
	table = []
	for subject in base[action]:
		for instance in base[action][subject]:
			for frame in base[action][subject][instance]:
				fframe = frame[:, coords].T.flat[:]
				table.append(fframe)
			#table.extend()
	table = pandas.DataFrame(table)
	return table

#Cria tabela com instancias por frames por cords
def formatTable(action, coords, n):
	alpha = 0.5
	table = []
	for subject in base[action]:
		for instance in base[action][subject]:
			#print(len(base[action][subject][instance]), end=' ')

			frames = [frame[:, coords].T.flat[:] for frame in base[action][subject][instance]]
			#padronizar nro de frames n
			a, b = math.ceil((len(frames)-n)/2), (len(frames)-n)//2
			#for i in range(1, table.size-1):
			#	stat, p = stats.kruskal(table.loc[i, :], table.loc[i-1, :])
			if b>0:
				frames = frames[a:-b]
			elif a>0:
				frames = frames[a:]
			table.extend(frames)
			#print(len(frames))
	table = pandas.DataFrame(table)
	return table

base = loadBase()
table = formatTable(1, [0, 1, 2], 31)
print(table)
exit(1)
'''
padronizar o nro de frames
alpha = 0.5
for i in range(1, table.size-1):
	stat, p = stats.kruskal(table.loc[i, :], table.loc[i-1, :])
	#print('Kruskal-Wallis: Estatisticas=%.3f, p=%.3f' % (stat, p))

	if p > alpha:
		print(i, 'Mesma distribuição')
	else:
		print(i, 'Distribução diferente')
'''

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


'''
tabela instancias por PCA features(2) frames
'''
# Separating out the features
x = table.loc[:, :].values


# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pandas.DataFrame(data = principalComponents
		   , columns = ['principal component 1', 'principal component 2'])
print(principalDf)

