import random
import string

stringAnswer = "morten_madsen*568154"
populationSize = 100
tourny_max = 10
Pe = 0.2
Pc = 0.7
Mc = 0.1
popList = []


def fitnessFunction(x):
    p = 0
    for i in range(len(stringAnswer)):
        if x[i] == stringAnswer[i]:
            p += 1
    return p / len(stringAnswer)


def mutate(child):
    child_list = list(child)
    if random.random() < Mc:
        mutation_point = random.randint(0, len(child_list) - 1)
        child_list[mutation_point] = random.choice(string.ascii_lowercase + string.digits + "*" + "_")
    return ''.join(child_list)


def generateFirstPopulation(popSize):
    population = []
    for _ in range(popSize):
        individual = ''.join(
            random.choice(string.ascii_lowercase + string.digits + "*" + "_") for _ in range(len(stringAnswer)))
        population.append(individual)
    return population


def evolveGeneration(popList):
    newPop = []
    while len(newPop) < len(popList):
        parent1, parent2 = selectParents(popList)
        child1, child2 = uniformCrossover(parent1, parent2, Pc)
        child1, child2 = mutate(child1), mutate(child2)
        newPop.extend([child1, child2])
    return newPop[:len(popList)]


def selectParents(popList):
    if random.random() < Pe:
        return sorted_population(popList)[:2]
    else:
        return tournament1(popList), tournament1(popList)


def tournament1(popList):
    tournament_pool = [random.choice(popList) for _ in range(tourny_max)]
    winner = max(tournament_pool, key=fitnessFunction)
    return winner


def sorted_population(popList):
    return sorted(popList, key=fitnessFunction, reverse=True)


def uniformCrossover(p1, p2, Pc):
    if random.random() < Pc:
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2


oldPop = generateFirstPopulation(populationSize)
genCounter = 0
working = True

while working:
    newPopulation = evolveGeneration(oldPop)
    best_individual = max(newPopulation, key=fitnessFunction)
    best_fitness = fitnessFunction(best_individual)
    print(f"Initial population {populationSize}, Generation {genCounter}, Crossover Probability: {Pc}, best try was: {best_individual}, with a fitness of {best_fitness:.4f}")
    if genCounter > 5000 or best_fitness >= 1:
        working = False
    oldPop = newPopulation
    genCounter += 1