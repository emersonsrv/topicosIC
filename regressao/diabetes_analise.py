from sklearn.datasets import load_diabetes
import pandas
import matplotlib.pyplot as plt

target = 'dp'

diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela[target] = diabetes.target

print(tabela.head(20))
f = open('diabetes20.txt', 'w')
f.write(str(tabela.head(20)))
f.close()
