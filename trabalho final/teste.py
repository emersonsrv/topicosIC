import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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
				table.append(frame[:, coords].T.flat[:])
	table = pandas.DataFrame(table)
	return table

#Cria tabela com instancias por frames por cords
def instancesByframesByCoords(action, coords):
    table = []
    for subject in base[action]:
        for instance in base[action][subject]:
            #print(len(base[action][subject][instance]), end=' ')
            
            row = [frame[:, coords].flat[:] for frame in base[action][subject][instance]]
            table.append(pandas.DataFrame(row))
    print()
    #table = pandas.DataFrame(table)
    return table