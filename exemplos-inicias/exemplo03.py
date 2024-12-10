import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# DADOS:
# Conjunto de dados fictício: lista de e-mails com rótulos 'spam' ou 'não spam' (ham)
data = {
    'email': [
        "Compre agora a oferta grátis",  # SPAM
        "Clique aqui para ganhar prêmio",  # SPAM
        "Reunião agendada para segunda",  # NÃO SPAM
        "Oferta exclusiva para você",  # SPAM
        "Vamos almoçar amanhã?",  # NÃO SPAM
        "Grátis: sua consulta médica",  # SPAM
        "Confirme sua inscrição no evento",  # NÃO SPAM
        "Grátis, clique aqui para baixar",  # SPAM
        "Atualização importante do projeto",  # NÃO SPAM
        "Oferta limitada, clique e ganhe",  # SPAM
    ],
    'label': [
        'spam', 'spam', 'não spam', 'spam', 'não spam',
        'spam', 'não spam', 'spam', 'não spam', 'spam'
    ]
}


# DATAFRAME = DF
df = pd.DataFrame(data)
df.head()


X = df['email']
y = df['label']

vetorizar = CountVectorizer()
X_vetorized = vetorizar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vetorized, y, test_size=0.3, random_state=42)



naive_bayes = MultinomialNB()

naive_bayes.fit(X_train, y_train)

naive_bayes



# Fazer Predição

y_pred = naive_bayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"A acuracia é de : {accuracy}")




novo_email = ["Estarei enviando um email para voce passar os dados do seu cartao"]
novo_email_vetorizado = vetorizar.transform(novo_email)
previsao = naive_bayes.predict(novo_email_vetorizado)
print(f"Predicado do novo email é: {previsao}")



# Matriz de Confusão

confusion_matrix(y_test, y_pred)



# relatório de classificação muito importante
classification_report(y_test, y_pred)