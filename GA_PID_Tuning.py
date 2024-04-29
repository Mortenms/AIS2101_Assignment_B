import matplotlib.pyplot as plt
import numpy as np
import random
import string
#pid params
Kp=0.0
Ki=0.0
Kd=0.0
PIDsetpoint=10
simDuration = 100
measuredValue = 0

#GA params
populationSize = 150
tourny_max = 10
Pe = 0.2
Pc = 0.7
Mc = 0.1
upperFloatLimit = 1
popList = []

#desired attributes
DesiredTs = 50
DesiredOS = 15
DesiredTr = 10
DesiredSSE = 0

#PID implement
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, measured_value):
        error = self.setpoint - measured_value

        P = self.Kp * error

        self.integral += error
        I = self.Ki * self.integral

        D = self.Kd * (error - self.prev_error)
        self.prev_error = error

        control_signal = P+I+D
        return control_signal
def sim(controller, duration, measuredVal):
    time = []
    output = []
    for t in range(duration):
        time.append(t)
        output.append(measuredVal)
        control_signal = controller.update(measuredVal)
        measuredVal += control_signal
    return output
def calculateSettlingTime(response, duration, setpoint):
    response_reversed = response.copy()[::-1]
    for i in range(duration): #
        if response_reversed[i] <= setpoint*0.95 or response_reversed[i] >= setpoint*1.05:
            return duration-i
def calculateOvershoot(response):
    peak = np.max(response) #finds max value of cs
    overshoot = (peak - response[-1]) / response[-1] * 100  # %OS
    return overshoot
def calculateRiseTime(response, duration, settleTime):
    flag1 = False
    flag2 = False
    for i in range(duration):
        if (response[settleTime] * 0.1) <= response[i] and flag1 == False:
            idx_lower = i
            flag1 = True
        if (response[settleTime] * 0.9) <= response[i] and flag2 == False:
            idx_upper = i
            flag2 = True
    x = idx_upper - idx_lower
    return x
def calculateSSE(response):
    return abs(PIDsetpoint - response[-1])
def calcAll(response, duration):
    Ts = calculateSettlingTime(response, duration, response[-1])
    OS = calculateOvershoot(response)
    Tr = calculateRiseTime(response, duration, Ts)
    SSE = calculateSSE(response)
    return Ts,OS,Tr,SSE

#GA implement

def fitnessFunction(response,duration):
    Ts, OS, Tr, SSE = calcAll(response,duration)
    x = 1 / ((DesiredTs - Ts)**2 + (DesiredOS-OS)**2 + (DesiredTr - Tr)**2 + (DesiredSSE-SSE)**2)
    return x

def generateFirstPopulation(popSize):
    population = []
    for _ in range(popSize):
        individual = [0, 0, 0]  # kp,ki,kd
        for i in range(3):
            individual[i] = random.uniform(0,upperFloatLimit)
            population.append(individual)
    return population

def mutate(child):
    if random.random() < Mc:
        child[random.randint(0,2)] = random.uniform(0,upperFloatLimit)
        return child
    return child


def sorted_population(popList,fitnessList):
    sorted_popList = sorted(popList, key=lambda individual: fitnessList[popList.index(individual)], reverse=True)
    return sorted_popList
def selectParents(popList,fitnessList):
    if random.random() < Pe:
        return sorted_population(popList,fitnessList)[:2]
    else:
        return tournament1(popList,fitnessList), tournament1(popList,fitnessList)


def tournament1(popList,fitnessList):
    tournament_pool = [random.choice(popList) for _ in range(tourny_max)]
    winner = max(tournament_pool, key=lambda individual: fitnessList[popList.index(individual)])
    return winner

def uniformCrossover(p1, p2, Pc):
    if random.random() < Pc:
        child1 = []
        child2 = []
        for _ in range(3):
            if random.random() < 0.5:
                child1.append(p1[_])
                child2.append(p2[_])
            else:
                child1.append(p2[_])
                child2.append(p1[_])
        return child1, child2
    return p1, p2

def evolveGeneration(popList,FitnessList):
    newPop = []
    for i in range(len(popList)):
        parent1, parent2 = selectParents(popList,FitnessList)
        child1, child2 = uniformCrossover(parent1, parent2, Pc)
        child1, child2 = mutate(child1), mutate(child2)
        newPop.extend([child1, child2])
    return newPop[:len(popList)]

def simResponseList(pop):
    list = []
    for i in range(len(pop)):
        list.append(sim(PIDController(pop[i][0],pop[i][1],pop[i][2],PIDsetpoint),simDuration,measuredValue))
    return list

def fitnessList(pop,responseList):
    list = []
    for i in range(len(pop)):
        list.append(fitnessFunction(responseList[i], simDuration))
    return list


#runtime vars
genCounter = 0
oldPop = generateFirstPopulation(populationSize)
working = True
controller = PIDController(Kp,Ki,Kd,PIDsetpoint)
responseList = []

#plotting
bestPidList = []
bestFitnessList = []

## RUNTIME
while working:
    responseList = simResponseList(oldPop)
    fitnessValues = fitnessList(oldPop,responseList)
    newPopulation = evolveGeneration(oldPop, fitnessValues)
    bestIndividual = max(newPopulation, key=lambda individual: fitnessValues[newPopulation.index(individual)])
    bestPidList.append(bestIndividual)
    bestFitness = fitnessFunction(sim(PIDController(bestIndividual[0],bestIndividual[1],bestIndividual[2],PIDsetpoint),simDuration,measuredValue),simDuration)
    bestFitnessList.append(bestFitness)
    print(f"Initial population {populationSize}, Generation {genCounter}, Mutation Probability {Mc}, "
          f"Crossover Probability: {Pc}, best try was: {bestIndividual}, with a fitness of {bestFitness:.4f}")
    if genCounter > 100 or bestFitness >= 1:
        working = False
    oldPop = newPopulation
    genCounter += 1


#PLOTTING
plt.figure(figsize=(10, 10))
ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
plotTime = np.arange(0, simDuration, 1)
plotGeneration = np.arange(0, genCounter, 1)

for i in range(len(bestPidList)):
    controller = PIDController(bestPidList[i][0], bestPidList[i][1], bestPidList[i][2], PIDsetpoint)
    response = sim(controller, simDuration, measuredValue)
    if response[-1] < 30: #prevents bigggg responses from plotting
        ax1.plot(plotTime, response),# label=f'Generation {i}')
    if i == len(bestPidList) - 1:  # Plot the last response on the second subplot
        ax2.plot(plotTime, response, label=f'Generation {i}')
        print(calcAll(response,simDuration))

plt.xlim(0, 120)
plt.ylim(0, 20)

ax1.set_title("Best PID response for each generation")
ax1.set_xlabel("Time")
ax1.set_ylabel("Output")
ax1.legend()

ax2.set_title("The last best PID response")
ax2.set_xlabel("Time")
ax2.set_ylabel("Output")
ax2.legend()

#fitness func over time
plt.subplot(3, 1, 3)
plt.plot(plotGeneration, bestFitnessList, marker='o')
plt.title("Development of Fitness Function")
plt.xlabel("Generation")
plt.ylabel("Fitness Value")

plt.tight_layout()
plt.show()

