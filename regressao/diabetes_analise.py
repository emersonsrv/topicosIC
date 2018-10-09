from sklearn.datasets import load_diabetes
import pandas
import matplotlib.pyplot as plt

target = 'dp'
# carrega a tabela de dados
diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela[target] = diabetes.target
# gera o grafico(.png) de correlacao entre os atributos IMC e DP
plt.scatter(tabela.bmi, tabela.dp)
plt.xlabel('IMC')
plt.ylabel('Dissease Prog.')
plt.savefig('correlacao-1.png')
plt.close()
# gera o grafico(.png) de correlacao entre os atributos S3 e DP
plt.scatter(tabela.s3, tabela.dp)
plt.xlabel('S3')
plt.ylabel('Dissease Prog.')
plt.savefig('correlacao-2.png')
plt.close()
# gera o grafico(.png) de correlacao entre os atributos S5 e DP
plt.scatter(tabela.s5, tabela.dp)
plt.xlabel('S5')
plt.ylabel('Dissease Prog.')
plt.savefig('correlacao-3.png')

# grava arquivo com database headset
print(tabela.head(20))
f = open('diabetes20.txt', 'w')
f.write(str(tabela.head(20)))
f.close()
