import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier # pip install catboost

# problema 
# aber se o solo é (0 - vivo, 1 - dano devido a outras causas, 2 = dano devido a pesticidas)
# coletando os dados
train_d = pd.read_csv('/content/treino.csv') # fazer o dowload do treino.csv

train_d.head(10) # mostra os 10 primeiros dados da tabela
train_d.info() # tarz os elementos da tabela e seus possiveis valores
train_d.describe() # mostra os dados estatísticos da tabela

# treinamento


train_d['Crop_Damage'].value_counts()

ax = sns.countplot(x=train_d['Season']) # faz um grafico em barra da variável alvo ou qualquer outra variável


# verifica se tem valor nulo esoma quantos valore nulos tem
train_d.isnull().sum()

# verifica se tem valor duplicado neste caso não tem valor nulo pois cada dado tem seu id própio
train_d.duplicated().sum()

# excluir á variável id, para mostrar valores duplicados
train_d.drop('ID', axis=1, inplace=True)

# agora verificamos se tem valores duplicados
train_d.duplicated().sum() # agora tem pois excluimos o id e cada elemento e como se fosse igual

# exclusao de valores ausentes com dropna

train_d.dropna(inplace=True)

train_d.describe() # traz o total dos valores de cada variável

# One Hot Criar variáveis númericas
for col in ['Crop_Type', 'Soil_Type', 'Pesticide_Use_Category', 'Season']:
  trein_d = pd.get_dummies(train_d, columns=[col])
  

train_d.describe()

# Separa os dados de treinamento e teste
X = train_d.drop(['Crop_Damage'], axis=1) 
y = train_d['Crop_Damage'].values.reshape(-1,1)

# Amostragem

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)


# treinamento da máquina preditiva

maquina_preditiva = CatBoostClassifier(n_estimators=1000, max_depth=4, random_state=23)
maquina_preditiva.fit(X_train, y_train)
predicoes = maquina_preditiva.predict_proba(X_test)


# avalia se  o modelo está bom

result = maquina_preditiva.score(X_test, y_test)
print("A minha acuracia é : %.f%%" % (result * 100.0))