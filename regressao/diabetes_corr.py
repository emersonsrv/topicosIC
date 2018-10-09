from sklearn.datasets import load_diabetes
import pandas
import matplotlib.pyplot as plt

target = 'dp'
# gera tabela de correlacao
diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela[target] = diabetes.target
# grava arquivo com a tabela de correlacao
print(tabela.corr())
f = open('diabetes_corr.txt', 'w')
f.write(str(tabela.corr()))
f.close()


