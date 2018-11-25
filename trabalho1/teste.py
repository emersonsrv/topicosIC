import pandas
import numpy as np

basePath = '../../MSRAction3D/skeleton/'

def loadData(action, subject, instance):
    ins = np.loadtxt(basePath + 'a%02i_s%02i_e%02i_skeleton.txt'%(action, subject, instance))
    ins = ins.reshape((ins.shape[0]//20, 20, 4))
    return ins

def framesByCoords(action, coords):
    table = []
    for subject in range(1, 11):
        for instance in [1, 2, 3]:
            try:
                data = loadData(action, subject, instance)
            except IOError:
                continue
            
            for frame in data:
                table.append(frame[:, coords].T.flat[:])
    table = pandas.DataFrame(table)
    return table