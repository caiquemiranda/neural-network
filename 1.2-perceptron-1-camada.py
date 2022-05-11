
# importação da biblioteca para otmização de calculos
import numpy as np

# definição das variáveis de entrada e pesos
entradas = np.array([0, 1])
pesos = np.array([0.5, 0.5])

# função de soma
def soma(entradas, pesos):
    # utilização do método dot para realizar o produto escalar
    return entradas.dot(pesos)

# chamada da função de soma com os valores de entrada e pesos
s = soma(entradas, pesos)

# função de ativação(step function)
def step_func(soma):
    if soma >= 1:
        return 1
    return 0

r = step_func(s)
