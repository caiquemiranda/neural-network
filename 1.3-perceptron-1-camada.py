# importação da biblioteca para otmização de calculos
import numpy as np

# definição das variáveis de entrada, saidas, pesos e taxa de aprendizagem

#entradas = np.array([[0,0],[0,1], [1,0], [1,1]])
#saidas = np.array([0,0,0,1])
#entradas = np.array([[0,0],[0,1], [1,0], [1,1]])
#saidas = np.array([0,1,1,1])

entradas = np.array([[0,0],[0,1], [1,0], [1,1]])
saidas = np.array([0,1,1,0])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

# função de ativação(step function)
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

# função de para calcular a saída
def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

# função de treinamento
def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('Peso atualizado: '+ str(pesos[j]))
        print('Total de erros: '+ str(erroTotal))

# chamada da função de treinamento
treinar()

# imprime a rede neural treinada
print('Rede neural treinada')

# imprime a saída da rede neural treinada
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))