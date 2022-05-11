
# definção das variáveis de entrada e pesos
entradas = [1, 7, 5]
pesos = [0.8, 0.1, 0]

# função de soma
def func_soma(entradas, pesos):
    soma = 0                                   # inicializa a soma
    for i in range(len(entradas)):
        # print(entradas[i], pesos[i])         # visualização dos valores
        soma += entradas[i] * pesos[i]
    return soma

# chamada da função de soma com os valores de entrada e pesos
soma = func_soma(entradas, pesos)

# função de ativação(step function)
def step_func(soma):
    if soma >= 1:
        return 1
    return 0


r = step_func(soma)