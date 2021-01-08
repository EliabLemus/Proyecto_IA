from enum import Enum
import random, math, itertools,pickle
import numpy as np
import NeuralNetwork 
# individuals_population=[]
max_generations = 10
global_population = 8
TRAIN = []
TEST = []
LAYERS = []

class Solution:
    def __init__(self, solution_proposed=[], fitnessValue=0):
        self.solution_proposed = solution_proposed
        self.fitnessValue = fitnessValue


class Row:
    def __init__(self, alph=0, lamb=0, max_iteration=0, keep_prob=0):
        self.alph = alph
        self.lamb = lamb
        self.max_iteration = max_iteration
        self.keep_prob = keep_prob


# Return an array of random values between low and high boundaries
def getSolutions(low, high, size):
    return [int(random.uniform(low, high)) for _ in range(size)]


# Return an array of arrays with random values
def getPopulation(size, n=3):
    """
    numberOfSolutions: n^2
    size: Size of population
    """
    numberOfSolutions = n ** 2
    population = []
    limit = n ** 2
    limit = limit + 1
    # print('limit: ', limit)
    for i in range(0, size):
        population.append(Solution(getSolutions(1, limit, numberOfSolutions)))
    return population


def printObject(o):
    tmp = vars(o)
    for item in tmp:
        print(item, ":", tmp[item])


def commonSum(n):
    return (n * (n ** 2 + 1)) / 2


def differencesSum(square_data=[], n=0):
    # suma diferencias: suma_fila1 - suma_comun + suma_col1 - suma_comun + suma_diag1 - suma_comun
    common = commonSum(n)
    rows = getRows(square=square_data, n=n)
    cols = getColumns(square=square_data, n=n)
    diag1 = getDiag(square=square_data, n=n)
    diag2 = getDiagInverse(square=square_data, n=n)
    sum_row = 0
    sum_col = 0
    sum_diag1 = 0
    sum_diag2 = 0
    for r in rows:
        sum_row += abs(sum(r) - common)

    for c in cols:
        sum_col += abs(sum(c) - common)

    sum_diag1 += abs(sum(diag1) - common)
    sum_diag2 += abs(sum(diag2) - common)

    return sum_row + sum_col + sum_diag1 + sum_diag2




def getDiag(square=[], n=0):
    diag = []
    filas = len(square) / n
    columnas = filas
    contador = 1
    for a in range(0, int(columnas)):
        indice = int((a * n) + contador)
        diag.append(square[indice - 1])
        contador += 1
    return diag


def getDiagInverse(square=[], n=0):
    diagInverse = []
    filas = len(square) / n
    columnas = filas
    contador = filas
    for a in range(0, int(columnas)):
        indice = int((a * n) + contador)
        diagInverse.append(square[indice - 1])
        contador -= 1
    return diagInverse

class Criteria(Enum):
    MAXIMA_GENERACION = 0
    MEJOR_VALOR = 1
    FITNES_PROMEDIO = 2


def getCriteriaName(criteria):
    if criteria == "max_generation":
        return "Maxima generacion"
    elif criteria == "best_value":
        return "Mejor valor"
    elif criteria == "fitness_average":
        return "fitness promedio"
    else:
        criteria = Criteria.MAXIMA_GENERACION


def check_criteria(generation, population=[]):
    """
    population: individuals with solution propossals
    generation: number of generation working
    criteria: Type of criteria to evaluate
    """

    # if generation >= max_generations:
    #     return True
    
    # else:
    #     # un miembro de la poblacion alcance un valor fitnes.
    #     population.sort(key=lambda x: x.fitnessValue, reverse=False)
    #     print("-> Generacion {}: {}".format(generation, population[0].fitnessValue))
    #     # print('minimo: ', population[0].fitnessValue)
    #     for i in population:
    #         if i.fitnessValue == 0:
    #             return True
    #         else:
    #             return False
    return generation >= max_generations


class Parents(Enum):
    TOURNAMENT = 0
    BEST_VALUE = 1
    PAIRS = 2


def getChooseFathersName(choose_father_option):
    if choose_father_option == "tournament":
        return "Torneo"
    elif choose_father_option == "best_value":
        return "Mejor valor"
    elif choose_father_option == "pairs":
        return "Pares"


def chooseFathers(population, choose_father_options="best_value"):
    """
    will be selected the best of two
    """

    if choose_father_options == "tournament":
        tipo = Parents.TOURNAMENT
    elif choose_father_options == "best_value":
        tipo = Parents.BEST_VALUE
    elif choose_father_options == "pairs":
        tipo = Parents.PAIRS
    else:
        tipo = Parents.TOURNAMENT
    parents = []
    # population.sort(key=lambda x: x.fitnessValue, reverse=False)
    # print(tipo.name)
    if tipo == Parents.TOURNAMENT:  # tournament
        # Seleccion por torneo
        # population.sort(key=lambda x: x.fitnessValue, reverse=False)
        limit = int(len(population) / 2)
        for i in range(0, limit):
            parentA = population[i]
            parentB = population[i + 1]
            parents.append(
                parentB if parentB.fitnessValue < parentA.fitnessValue else parentA
            )
            i += 2
        return parents
    elif tipo == Parents.BEST_VALUE:  # Best value
        # padres con el mejor valor fitness
        population.sort(key=lambda x: x.fitnessValue, reverse=True)
        limit = int(len(population) / 2)
        for i in range(0, limit):
            parentA = population[i]
            parents.append(parentA)
        return parents
    elif tipo == Parents.PAIRS:
        for j in range(0, len(population)):
            if j % 2 == 0:
                parentB = population[j]
                parents.append(parentB)

        return parents


def match(parents):
    """
    Emparejar
    """
    mid = int(len(parents) / 2)
    sons = []
    i = 0
    while i < mid:
        son1 = Solution()
        son1.solution_proposed = cross(
            parents[i].solution_proposed, parents[mid + i].solution_proposed
        )
        son1.solution_proposed = mutate(son1.solution_proposed)
        son1.fitnessValue = fitnessValue(square_data=son1.solution_proposed)

        son2 = Solution()
        son2.solution_proposed = cross(
            parents[mid + i].solution_proposed, parents[i].solution_proposed
        )
        son2.solution_proposed = mutate(son2.solution_proposed)
        son2.fitnessValue = fitnessValue(square_data=son2.solution_proposed)

        sons.append(son1)
        sons.append(son2)
        i += 1

    parents = sorted(parents, key=lambda item: item.fitnessValue, reverse=True)

    i = 0
    new_population = []
    while i < len(parents):
        new_population.append(sons[i])
        new_population.append(parents[i])
        i += 1

    return new_population


def cross(parent1, parent2):
    """
    Cruzar
    """
    value1 = random.uniform(0,1)
    value2 = random.uniform(0,1)
    value3 = random.uniform(0,1)
    value4 = random.uniform(0,1)
    
    
    w1 = parent1[0] if value1 <= 0.5 else parent2[0]
    w2 = parent1[1] if value2 <= 0.5 else parent2[1]
    w3 = parent1[2] if value3 <= 0.5 else parent2[2]
    w4 = parent1[3] if value4 <= 0.5 else parent2[3]

    return [w1,w2,w3,w4]


def mutate(solution):
    
    position = random.randint(0,3)
    solution[position] = random.randint(1,10)
    return solution

def fitnessValue(square_data=[]):
    """
    Exactitud de la validacion de la red neuronal
    """
    #TODO: Evaluar la red neuronal para obtener el valor fitness. 
    # n = math.sqrt(len(square_data))
    # X = square_data
    # Y = [(x, len(list(y))) for x, y in itertools.groupby(X)]
    # # Repetidos:
    # repeated = 0
    # for a in Y:
    #     if a[1] > 1:
    #         repeated += 1
    # differences = differencesSum(square_data=square_data, n=n)
    # result = (1 + repeated) * differences + (repeated ** 2)
    hyper_p = NeuralNetwork.getHyperParemeters(setup=square_data)
    print('setup: ', square_data)
    print('hyper_p:',hyper_p)
    training, test, model = NeuralNetwork.useNetwork(TRAIN,TEST,LAYERS,alpha=hyper_p[0], iterations=hyper_p[2], lambd=hyper_p[1], keep_prob=hyper_p[3])
    return test


if __name__ == "__main__":
    NeuralNetwork.buildHyperParameters()
    TRAIN,TEST,LAYERS=NeuralNetwork.initNeuralNetwork()
    items = np.random.randint(1,10,size=(global_population,4))
    print(items)
    population = []
    for i in items:
        fit_value = fitnessValue(square_data=i)
        population.append(Solution(solution_proposed=i,fitnessValue=fit_value))
    generation = 0

    stop = check_criteria(generation, population=population)
    while stop != True:
        print('generation:', generation)
        print("\n")
        # choose parents:
        parents = chooseFathers(population, choose_father_options="best_value")
        population = match(parents)
        generation += 1
        stop = check_criteria(generation, population=population)

    population.sort(key=lambda x: x.fitnessValue, reverse=True)
    print("\n")
    
    
    
    with open('TrainedModels/last_population.dat', 'wb') as f:
        pickle.dump(population,f)

    hyper_p = NeuralNetwork.getHyperParemeters(setup=population[0].solution_proposed)
    print('Best model details:')
    print('parameters:', hyper_p)
    print('alpha:', hyper_p[0])
    print('lambda:', hyper_p[1])
    print('max_iteration:', hyper_p[2])
    print('keep_prob:', hyper_p[3])
    print(population[0].solution_proposed)
    print('fitness value:', population[0].fitnessValue)
    training, test, model = NeuralNetwork.useNetwork(TRAIN,TEST,LAYERS,alpha=hyper_p[0], iterations=hyper_p[2], lambd=hyper_p[1], keep_prob=hyper_p[3])
    with open('TrainedModels/best_model.dat', 'wb') as f:
        pickle.dump(model,f)