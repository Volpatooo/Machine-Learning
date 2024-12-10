import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


# Conjunto de dados fictício corrigido com mais exemplos de "não spam" (ham)
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
        "Ganhe prêmios clicando aqui",  # SPAM
        "Não perca a chance de ganhar prêmios",  # SPAM
        "Consulta médica gratuita, clique para agendar",  # SPAM
        "Ganhe uma viagem para o Caribe, só hoje",  # SPAM
        "Quer um desconto exclusivo? Clique aqui",  # SPAM
        "Oferta imperdível para você, clique aqui",  # SPAM
        "Inscreva-se agora e ganhe prêmios",  # SPAM
        "Ofertas especiais só hoje, aproveite",  # SPAM
        "Aproveite as ofertas relâmpago, clique aqui",  # SPAM
        "Você ganhou um presente exclusivo, clique para resgatar",  # SPAM
        "Sua chance de ganhar um prêmio está aqui",  # SPAM
        "Promoção imperdível, só até hoje",  # SPAM
        "Última chance de ganhar dinheiro grátis",  # SPAM
        "Ganhe um prêmio agora mesmo, só por clicar",  # SPAM
        "Você foi selecionado para uma oferta exclusiva",  # SPAM
        "Ofertas incríveis esperando por você",  # SPAM
        "Ganhe ingressos para o evento exclusivo",  # SPAM
        "Não perca a chance de ganhar um prêmio em dinheiro",  # SPAM
        "Ganhe até 1000 reais em prêmios, clique agora",  # SPAM
        "Você ganhou um vale-presente de R$100",  # SPAM
        "Ganhe um carro novo com este simples clique",  # SPAM
        "Ofertas limitadas, aproveite agora",  # SPAM
        "Clique e ganhe um prêmio instantâneo",  # SPAM
        "Participe da nossa promoção e ganhe prêmios",  # SPAM
        "Desconto exclusivo para você, clique aqui",  # SPAM
        "Ganhe uma viagem para a Europa",  # SPAM
        "Apenas hoje: prêmio em dinheiro por se cadastrar",  # SPAM
        "Oferta exclusiva para novos membros",  # SPAM
        "Você foi escolhido para ganhar uma grande recompensa",  # SPAM
        "Promoção de Natal: prêmios exclusivos",  # SPAM
        "Oferta limitada de 50% de desconto, clique aqui",  # SPAM
        "Ganhe uma assinatura grátis de 3 meses",  # SPAM
        "Seu prêmio está esperando por você",  # SPAM
        "Participe e ganhe um smartphone novo",  # SPAM
        "Clique para saber como ganhar prêmios agora mesmo",  # SPAM
        "Promoção de verão, ganhe prêmios incríveis",  # SPAM
        "Última chance: 90% de desconto, só hoje",  # SPAM
        "Ganhe prêmios instantâneos agora!",  # SPAM
        "Ganhe uma linha de crédito de R$5.000",  # SPAM
        "Promoção de última hora: desconto em viagem",  # SPAM
        "Você tem um cupom de desconto esperando por você",  # SPAM
        "Ganhe um vale-compras de R$200",  # SPAM
        "Ganhe um prêmio incrível em dinheiro, clique aqui",  # SPAM
        "Esta é sua chance de ganhar uma viagem",  # SPAM
        "Você ganhou um prêmio em nossa promoção",  # SPAM
        "Participe da promoção e ganhe prêmios incríveis",  # SPAM
        "Ganhe até R$1.000 em dinheiro grátis",  # SPAM
        "Você foi escolhido para um prêmio especial",  # SPAM
        "Clique agora e ganhe um desconto de 70%",  # SPAM
        "Ganhe ingressos para o cinema",  # SPAM
        "Ganhe prêmios todos os dias, só por clicar",  # SPAM
        "Seu prêmio está garantido, clique para resgatar",  # SPAM
        "Aproveite descontos exclusivos por tempo limitado",  # SPAM
        "Ganhe ingressos para shows",  # SPAM
        "Desconto de 50% em produtos de tecnologia",  # SPAM
        "Ganhe um prêmio por participar da nossa pesquisa",  # SPAM
        "Sua chance de ganhar um prêmio exclusivo",  # SPAM
        "Participe da nossa rifa e ganhe um prêmio incrível",  # SPAM
        "Última chamada para participar da nossa promoção",  # SPAM
        "Ganhe dinheiro em sua conta bancária, clique aqui",  # SPAM
        "Oportunidade única para ganhar prêmios",  # SPAM
        "Participe da nossa promoção e ganhe uma viagem para Paris",  # SPAM
        "Ganhe um prêmio por responder essa pesquisa",  # SPAM
        "Ganhe prêmios incríveis com um clique",  # SPAM
        "Ganhe ingressos para eventos exclusivos",  # SPAM
        "Desconto especial para você, clique agora",  # SPAM
        "Clique aqui e ganhe um bônus de R$200",  # SPAM
        "Receba um bônus agora mesmo por se inscrever",  # SPAM
        "Ganhe uma assinatura gratuita por um mês",  # SPAM
        "Você ganhou um prêmio instantâneo!",  # SPAM
        "Últimos dias para ganhar prêmios incríveis",  # SPAM
        "Participe da nossa promoção e ganhe um iPhone",  # SPAM
        "Ganhe prêmios e descontos exclusivos",  # SPAM
        "Ganhe ingressos para o evento de sua escolha",  # SPAM
        "Ganhe uma viagem para o Caribe, participe agora",  # SPAM
        "Oportunidade única para você ganhar dinheiro",  # SPAM
        "Você foi escolhido para um prêmio incrível",  # SPAM
        "Ganhe um voucher de R$100, clique aqui",  # SPAM
        "Ganhe prêmios incríveis todos os dias",  # SPAM
        "Clique aqui para saber como ganhar prêmios",  # SPAM
        "Ganhe um carro novo, clique aqui",  # SPAM
        "Última chance de ganhar prêmios fantásticos",  # SPAM
        "Clique para saber como ganhar uma casa",  # SPAM
        "Receba um prêmio de R$500 agora mesmo",  # SPAM
        "Promoção imperdível, não perca!",  # SPAM
        "Apenas hoje: desconto exclusivo em todos os produtos",  # SPAM
        "Ganhe um prêmio por participar da nossa pesquisa",  # SPAM
        "Ganhe uma viagem para qualquer lugar do mundo",  # SPAM
        "Ganhe um bônus por se inscrever na nossa plataforma",  # SPAM
        "Clique aqui para ganhar prêmios exclusivos",  # SPAM
        "Ganhe ingressos para o evento de sua escolha",  # SPAM
        "Ganhe um vale-presente agora mesmo",  # SPAM
        "Clique e descubra como ganhar prêmios agora",  # SPAM
        "Promoção exclusiva para você",  # SPAM
        "Ganhe uma viagem para qualquer destino turístico",  # SPAM
        "Ganhe dinheiro com esta promoção",  # SPAM
        "Clique aqui e ganhe descontos em grandes marcas",  # SPAM
        "Oferta exclusiva: ganhe prêmios incríveis",  # SPAM
        "Promoção de Natal, ganhe prêmios agora",  # SPAM
        "Ganhe um prêmio todos os dias, clique aqui",  # SPAM
        "Última chance de ganhar dinheiro grátis",  # SPAM
        "Ganhe prêmios instantâneos com este clique",  # SPAM
        "Ofertas incríveis esperando por você",  # SPAM
        "Participe da nossa promoção e ganhe prêmios incríveis",  # SPAM
        "Ganhe ingressos para shows e eventos",  # SPAM
        "Não perca a chance de ganhar um prêmio especial",  # SPAM
        "Ganhe uma viagem de luxo para o exterior",  # SPAM
        "Aproveite a promoção e ganhe um prêmio agora",  # SPAM
        "Ganhe um prêmio em dinheiro agora",  # SPAM
        "Você ganhou um prêmio por participar da nossa pesquisa",  # SPAM
        "Oferta especial para novos cadastros",  # SPAM
        "Ganhe um smartphone novinho",  # SPAM
        "Ganhe até 1000 reais clicando aqui",  # SPAM
        "Clique e descubra o prêmio exclusivo esperando por você",  # SPAM
        "Ganhe ingressos para o evento de sua escolha",  # SPAM
        "Ganhe um prêmio por participar da nossa pesquisa",  # SPAM
        "Participe da promoção e ganhe prêmios exclusivos",  # SPAM
        "Ganhe uma linha de crédito de até R$10.000",  # SPAM
        "Participe agora e ganhe uma viagem grátis",  # SPAM
        "Ofertas incríveis esperando por você",  # SPAM
        "Ganhe ingressos para eventos exclusivos",  # SPAM
        "Clique para saber como ganhar prêmios incríveis",  # SPAM
        "Aproveite as ofertas incríveis só hoje",  # SPAM
        "Ganhe ingressos para o evento mais esperado do ano",  # SPAM
        "Última chance de ganhar grandes prêmios",  # SPAM
        "Ganhe um prêmio valioso agora mesmo",  # SPAM
        "Você ganhou um prêmio de até R$10.000",  # SPAM
        "Oferta especial para você, clique agora",  # SPAM
        "Ganhe prêmios todos os dias, não perca!",  # SPAM
        "Participe da nossa promoção e ganhe uma casa",  # SPAM
        "Reunião agendada para quinta-feira",  # NÃO SPAM
        "Aula de yoga cancelada para amanhã",  # NÃO SPAM
        "Reunião com cliente marcada para as 10h",  # NÃO SPAM
        "Novo projeto de marketing lançado hoje",  # NÃO SPAM
        "Estamos esperando por você na conferência",  # NÃO SPAM
        "Confirmar presença na reunião de amanhã",  # NÃO SPAM
        "Envio do relatório financeiro mensal",  # NÃO SPAM
        "Atualização do cronograma do projeto",  # NÃO SPAM
        "E-mail de boas-vindas à nossa plataforma",  # NÃO SPAM
        "Relatório de vendas de outubro",  # NÃO SPAM
        "Aprovamos sua solicitação de reembolso",  # NÃO SPAM
        "Você tem um novo arquivo compartilhado",  # NÃO SPAM
        "Almoço de negócios agendado para sexta-feira",  # NÃO SPAM
        "Convite para entrevista de emprego",  # NÃO SPAM
        "Envio de documento solicitado",  # NÃO SPAM
        "Novo curso online disponível para matrícula",  # NÃO SPAM
        "Detalhes do evento de networking",  # NÃO SPAM
        "Sua inscrição foi confirmada no seminário",  # NÃO SPAM
        "Novo artigo disponível no nosso blog",  # NÃO SPAM
        "Confirmar horários de reuniões para a próxima semana",  # NÃO SPAM
        "Relatório de desempenho mensal",  # NÃO SPAM
        "Novo pedido foi enviado com sucesso",  # NÃO SPAM
        "Agenda de eventos corporativos do mês",  # NÃO SPAM
        "Seu pedido foi processado com sucesso",  # NÃO SPAM
        "Obrigado por sua inscrição no curso",  # NÃO SPAM
        "Reunião para definir estratégias de marketing",  # NÃO SPAM
        "Alterações no horário da reunião",  # NÃO SPAM
        "Solicitação de licença aprovada",  # NÃO SPAM
        "Resumo do encontro de hoje",  # NÃO SPAM
        "Detalhes sobre o evento de lançamento",  # NÃO SPAM
        "Envio de contrato para assinatura",  # NÃO SPAM
        "Novo artigo disponível para leitura",  # NÃO SPAM
        "Mudança de local para a reunião de amanhã",  # NÃO SPAM
    ],
    'label': [
        'spam', 'spam', 'não spam', 'spam', 'não spam',
        'spam', 'não spam', 'spam', 'não spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'spam', 'spam', 'spam', 'spam', 'spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
        'não spam', 'não spam', 'não spam', 'não spam', 'não spam',
    ]
}



df = pd.DataFrame(data)
df.head()


# Faze de treinamento

X = df['email']
y = df['label']

vetorizar = CountVectorizer()
X_vetorized = vetorizar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vetorized, y, test_size=0.3, random_state=42)



nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
Y_pred_nb = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"A acuracia é de : {accuracy}")




rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"A acuracia é de : {accuracy}")




novo_email = ["Seu produto chegará em dois dias passe seus dados do cartão"]
novo_email_vetorizado = vetorizar.transform(novo_email)

previsao_rf = rf_classifier.predict(novo_email_vetorizado)
previsao_nb = nb_classifier.predict(novo_email_vetorizado)
previsao_clf = clf
print(f"Previsão De Árvore do novo email é: {previsao_rf}")
print(f"Previsão De Nb do novo email é: {previsao_nb}")