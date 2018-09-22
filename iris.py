#dados
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import numpy
import matplotlib.pyplot as plt
import time
import sys

# caracteristicas passadas por parametro [0, 3]
c1 = int(sys.argv[1])
c2 = int(sys.argv[2])

iris = datasets.load_iris()
#X = iris.data[:, 0:3]  # usando 2, 3
X = numpy.array((iris.data[:,c1],iris.data[:, c2])).T
#X = numpy.array((iris.data[:,0],iris.data[:, 1],iris.data[:, 3])).T
y = iris.target #classificacao
#0 Comprimento da sépala; 1 Largura da sépala; 2 comprimento da pétala; Largura da pétala 
#setosa, versicolor, virginica

clf = KNeighborsClassifier(n_neighbors=3)
#clf = MLPClassifier(alpha=0.01, max_iter=2000)
#dados de treinamento 'até 40' de cada classe
xt = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
yt = numpy.concatenate([y[:40], y[51:90], y[101:140]])

ini = time.clock()
clf.fit(xt, yt)  
print('fit time:', time.clock()-ini)

#validacão com o restante dos dados
xv = numpy.concatenate([X[40:50,:], X[90:100,:], X[140:150,:]])
yv = numpy.concatenate([y[40:50], y[90:100], y[140:150]])

yp = clf.predict(xv)

print(yp)
print(yv)
print(confusion_matrix(yv,yp))  
print(classification_report(yv,yp))


