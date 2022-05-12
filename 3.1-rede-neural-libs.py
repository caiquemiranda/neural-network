# uso da biblioteca sklearn com dataset iris

# importando bibliotecas
from sklearn.neural_network import MLPClassifier     # importando biblioteca com a rede neural
from sklearn.datasets import load_iris               # importando biblioteca com o dataset iris


# definindo os parametros da rede neural
# carregando o dataset iris
iris = load_iris() 
iris                                 
entradas = iris.data
saidas = iris.target

# criando o modelo da rede neural

nn_model = MLPClassifier(verbose=True,                     # imprime as alterações do processo de treinamento(err, inter, ...)
                         max_iter=1000,                    # numero de iterações
                         tol=0.00001,                      # tolerancia para convergência
                         activation='relu',                # função de ativação
                         learning_rate_init=0.0001)        # taxa de aprendizado   
nn_model.fit(entradas, saidas)

# testando o modelo
# previsão com um novo registro

nn_model.predict([[5, 7.2, 5.1, 2.2]])