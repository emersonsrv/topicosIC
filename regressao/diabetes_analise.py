from sklearn.datasets import load_diabetes
import pandas
import matplotlib.pyplot as plt

target = 'dp'

diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela[target] = diabetes.target

plt.scatter(tabela.bmi, tabela.dp)
plt.xlabel('IMC')
plt.ylabel('Dissease Prog.')
plt.savefig('correlacao-1.png')
plt.close()

plt.scatter(tabela.s3, tabela.dp)
plt.xlabel('S3')
plt.ylabel('Dissease Prog.')
plt.savefig('correlacao-2.png')
plt.close()

plt.scatter(tabela.s5, tabela.dp)
plt.xlabel('S5')
plt.ylabel('Dissease Prog.')
plt.savefig('correlacao-3.png')


print(tabela.head(20))
f = open('diabetes20.txt', 'w')
f.write(str(tabela.head(20)))
f.close()
