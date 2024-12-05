# importação de bibliotecas
# pip install numpy
#

import numpy as np
import pandas as pd

# gerar um dataset
np.random.seed(0)
n_samples = 10000

# horas estudo
horas_estudo = np.random.uniform(0, 24, n_samples)

# Cansado => 1,  Não Cansado => 0
cansado = np.random.choice([0, 1], n_samples)

# notas = Variável alvo
nota = np.minimum(horas_estudo * np.random.uniform(3,5), 100) - (cansado * 10)

nota
np.sort(nota)

# garantir os valores corretos
nota = np.clip(nota, 0, 100)
np.sort(nota)

data = pd.DataFrame({
    'Horas de Estudo':horas_estudo,'Cansado':cansado,'Nota':nota
    })

data.head()


# treinamento
# dividir treino e teste

# x =  todas as variáveis exceto a variável alvo(nota)
x = data[['Horas de Estudo', 'Cansado']]

# y = somente a váriavel alvo
y = data['Nota']
# criar variáveis de treino e de teste
X_train, X_teste, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)