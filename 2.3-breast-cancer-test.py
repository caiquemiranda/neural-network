# importando as bibliotecas necessárias
import numpy as np
from sklearn import datasets

# definindo funções
# definindo a função de ativação sigmoid
def sigmoid(soma):
    return 1 / ( 1 + np.exp ( -soma ) )      # formula função sigmoid


# derivada da função sigmoid
def derivadaSigmoid(sig):
    return sig * (1 - sig)

# parametros da rede
momentum = 1
taxaAprendizagem = 0.6
epocas = 1_000

# inicializando pesos aleatoriamente
pesos0 = 2 * np.random.random((30, 3)) - 1
pesos1 = 2 * np.random.random((3, 1)) - 1

# dataset_breast_cancer
base = datasets.load_breast_cancer()

# entradas
entradas = base.data

# resultados esperados(classes)
valoresSaida = base.target
saidas = np.empty([569, 1], dtype=int)
for i in range(569):
    saidas[i] = valoresSaida[i]

# treinamento running time
for i in range(epocas):
    # camada de entrada
    camadaEntrada = entradas                                             # variavel auxiliar para camada de entrada
    
    # primeira camada
    somaSinapse0 = np.dot(camadaEntrada, pesos0)                         # produto escalar entradas e pesos0
    camadaOculta = sigmoid(somaSinapse0)                                 # função sigmoid para camada oculta
    
    # segunda camada
    somaSinapse1 = np.dot(camadaOculta, pesos1)                          # produto escalar camada oculta e pesos1
    camadaSaida = sigmoid(somaSinapse1)                                  # função sigmoid para camada de saída
    
    # erro 
    erroCamadaSaida = saidas - camadaSaida                               # erro da camada de saída
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))                     # média absoluta do erro
    print("Erro: " + str(mediaAbsoluta))                                 # imprimindo erro

    derivadaSaida = derivadaSigmoid(camadaSaida)                         # derivada da função sigmoid para camada de saída
    deltaSaida = erroCamadaSaida * derivadaSaida                         # delta para camada de saída

    # backpropagation
    pesos1Transposta = pesos1.T                                          # matriz transposta de pesos1 para fazer a multiplicação de matrizes
    deltaSaidaPeso = deltaSaida.dot(pesos1Transposta)                    # produto escalar deltaSaida e pesos1Transposta
    deltaCamadaOculta = deltaSaidaPeso * derivadaSigmoid(camadaOculta)   # delta para camada oculta

    camadaOcultaTransposta = camadaOculta.T                              # matriz transposta de camada oculta para fazer a multiplicação de matrizes
    pesosnovo1 = camadaOcultaTransposta.dot(deltaSaida)                  # produto escalar camada oculta e deltaSaida
    pesos1 = (pesos1 * momentum) + (pesosnovo1 * taxaAprendizagem)       # ajustando pesos1

    camadaEntradaTransposta = camadaEntrada.T                             # matriz transposta de camada de entrada para fazer a multiplicação de matrizes
    pesosnovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)           # produto escalar camada de entrada e deltaCamadaOculta
    pesos0 = (pesos0 * momentum) + (pesosnovo0 * taxaAprendizagem)        # ajustando pesos0
    