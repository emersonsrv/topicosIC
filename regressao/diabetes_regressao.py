from sklearn.datasets import load_diabetes
import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

target = 'disease_progression'
attribute1 = 'age'
attribute2 = 'sex'

diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela[target] = diabetes.target


#seleciona duas colunas
X = tabela[[attribute1, attribute2]]

#separa em dois conjuntos, um para treinamento e outro para validação (20 últimos)
X_t = X[:-20]
X_v = X[-20:]
y_t = tabela[target][:-20]
y_v = tabela[target][-20:]

regr = linear_model.LinearRegression()

# treina o modelo
regr.fit(X_t, y_t)

# faz a predição
y_pred = regr.predict(X_v)

# coeficientes a
a = regr.coef_
print('Coeficientes: \n', a)
#intercepto b
b = regr.intercept_
print('Coeficientes: \n', b)
#y = 5.10*RM + -0.65*LSTAT + -1.24

#prediz manualmente os valores com base nos coeficientes encontrados na regressao
y_teste = a[0]*X_v[attribute1] + a[1]*X_v[attribute2] + b

#exibe o valor predito manualmente y_teste, que começa de 486
#exibe o valor real y_t
#exibe o valor predito pela regressão linear

print(y_teste[len(X)-20], y_t[0],y_pred[0])

#plota todos os valores de validação
plt.scatter(X_v[attribute2], y_v,  color='black')
plt.scatter(X_v[attribute2], y_pred, color='blue')
plt.legend(["Real", "Predito"])
plt.show()
