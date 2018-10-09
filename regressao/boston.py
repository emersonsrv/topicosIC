from sklearn.datasets import load_boston
import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

boston = load_boston()
tabela = pandas.DataFrame(boston.data)
tabela.columns = boston.feature_names
tabela['Preço'] = boston.target

#seleciona duas colunas
X = tabela[["RM", "LSTAT"]]

#separa em dois conjuntos, um para treinamento e outro para validação (20 últimos)
X_t = X[:-20]
X_v = X[-20:]
#print(X_t["RM"])
y_t = tabela["Preço"][:-20]
y_v = tabela["Preço"][-20:]

regr = linear_model.LinearRegression()

# treina o modelo
regr.fit(X_t, y_t)

# faz a predição
y_pred = regr.predict(X_v)

# coeficientes a
print('Coeficientes: \n', regr.coef_)
#intercepto b
print('Coeficientes: \n', regr.intercept_)
#y = 5.10*RM + -0.65*LSTAT + -1.24

#prediz manualmente os valores com base nos coeficientes encontrados na regressao
y_teste = 5.10*X_v["RM"] - 0.65*X_v["LSTAT"]- 1.24

#exibe o valor predito manualmente y_teste, que começa de 486
#exibe o valor real y_t
#exibe o valor predito pela regressão linear

print(y_teste[486], y_t[0],y_pred[0])

#plota todos os valores de validação
plt.scatter(X_v["LSTAT"], y_v,  color='black')
plt.scatter(X_v["LSTAT"], y_pred, color='blue')
plt.legend(["Real", "Predito"])
plt.show()
