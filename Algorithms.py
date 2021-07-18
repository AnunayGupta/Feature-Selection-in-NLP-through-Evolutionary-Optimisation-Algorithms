def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class PSO:
    def __init__(self, size, popu_size, model, maxEpoch=100, maxV=4, minV=-4, stopE=35, particles_init=None):
        self.size = size
        if particles_init is not None:
            self.particles = particles_init
        else:
            self.particles = np.random.randint(2, size=(popu_size, self.size))
        self.maxEarlyStop = stopE
        self.maxEpochs = maxEpoch
        self.maxVelocity = maxV
        self.minVelocity = minV
        self.popuSize = popu_size
        self.velocity = np.random.rand(popu_size, self.size)
        self.pbest = np.zeros((popu_size, self.size))
        self.pbestScore = np.zeros((popu_size,), dtype=float)
        self.model = model
        self.gbestScore = 0
        self.gbest = np.ones((self.size,), dtype=float)
        self.gbestPopu = np.ones((popu_size, self.size))

    def checkFitness(self, trainFeatures, valFeatures, trainLabels, valLabels):
        bestItr = np.zeros((self.size,))
        bestItrScore = 0
        for i in range(self.popuSize):
            reducedFeaturesTrain = reduceFeature(self.particles[i], trainFeatures)
            reducedFeaturesVal = reduceFeature(self.particles[i], valFeatures)
            self.model.fit(reducedFeaturesTrain, trainLabels)
            score = self.model.score(reducedFeaturesVal, valLabels)

            if score > self.pbestScore[i]:
                self.pbestScore[i] = score
                self.pbest[i] = self.particles[i]

            if score > self.gbestScore:
                self.gbestScore = score
                self.gbest = self.pbest[i]
                self.gbestPopu = self.pbest

            if score > bestItrScore:
                bestItrScore = score
                bestItr = self.particles[i]

        return bestItr, bestItrScore

    def updateParticles(self, itr, bestItr):
        wmax = 1.0
        wmin = 0.6
        kmin = 1.5
        kmax = 4.0
        w = wmax - (itr / self.maxEpochs) * (wmax - wmin)
        k = kmin + (itr / self.maxEpochs) * (kmax - kmin)
        for i in range(self.popuSize):
            self.velocity[i] = w * self.velocity[i] + k * np.random.rand() * (self.pbest[i] - self.particles[i]) + \
                               k * np.random.rand() * (self.gbest - self.particles[i]) + \
                               k * np.random.rand() * (bestItr - self.particles[i])

            for j in range(self.size):
                if self.velocity[i][j] > self.maxVelocity:
                    self.velocity[i][j] = self.maxVelocity
                elif self.velocity[i][j] < self.minVelocity:
                    self.velocity[i][j] = self.minVelocity

            veloSig = sigmoid(self.velocity[i])
            for j in range(self.size):
                self.particles[i][j] = (np.random.rand() < veloSig[j]) * 1

    def logging(self, allScores):
        print("Best Score ", self.gbestScore)
        print("Best Features: ", self.gbest)
        gbest_indices = np.where(self.gbest == 1)[0]
        gbest_num_elements = gbest_indices.shape[0]
        print("Indices Selected: ", gbest_indices)
        print("Total Elements Selected: ", gbest_num_elements)
        plt.plot(allScores, )
        plt.show()

    def saveLog(self, path):
      if not os.path.exists(path+"/"+ type(self.model).__name__):
        os.mkdir(path+"/"+ type(model).__name__)
      np.save(path+"/"+ type(self.model).__name__ + "/populationPSO.npy", self.particles)
      np.save(path+"/"+ type(self.model).__name__ + "/velocityPSO.npy", self.velocity)
      np.save(path+"/"+ type(self.model).__name__ + "/personalBestPSO.npy", self.pbest)
      np.save(path+"/"+ type(self.model).__name__ + "/globalBestPSO.npy", self.gbest)
      np.save(path+"/"+ type(self.model).__name__ + "/globalBestPopulationPSO.npy", self.gbestPopu)
      return

    def mainPso(self, trainFeatures, valFeatures, trainLabels, valLabels, path):
        allScores = []
        epoch = 0
        earlyStop = 0
        while epoch < self.maxEpochs:
            start = time()
            print(f"Epochs: {epoch + 1}/{self.maxEpochs}")
            bestItr, bestItrScore = self.checkFitness(trainFeatures, valFeatures, trainLabels, valLabels)
            self.updateParticles(itr=epoch, bestItr=bestItr)
            end = time()
            if allScores:
                if self.gbestScore == allScores[-1]:
                    earlyStop += 1
                    if earlyStop == self.maxEarlyStop:
                        print("Early Stopping due to no increase in fitness...")
                        self.logging(allScores)
                        self.saveLog(path)
                        return allScores
                else:
                    earlyStop = 0

            print(f"Time Taken: {round((end - start), 3)} secs\t Best Score: "
                  f"{self.gbestScore}\t Best Score Epoch: {bestItrScore}")
            self.saveLog(path)
            allScores.append(self.gbestScore)
            epoch += 1

        self.logging(allScores)
        return allScores

        class GA:
          def __init__(self, featuresTrain, labelsTrain, featureTest, labelsTest, model, nPopulation=30, nParentsMating=4,
                        nMutation=3, nosGen=100, newPopu=None):
            self.featuresTrain = featuresTrain
            self.labelsTrain = labelsTrain
            self.featuresTest = featureTest
            self.labelsTest = labelsTest
            self.nosPopu = nPopulation
            self.nosParentsMating = nParentsMating
            self.nosMutation = nMutation
            popuShape = (self.nosPopu, self.featuresTrain.shape[1])
            if newPopu is not None:
                self.population = newPopu
            else:
                self.population = np.random.randint(low=0, high=2, size=popuShape)

            self.numGenerations = nosGen
            self.model = model
            self.parents = np.empty(shape=(self.nosParentsMating, self.featuresTrain.shape[1]))
            offspringSize = (self.nosPopu - self.nosParentsMating, self.featuresTrain.shape[1])
            self.offspring = np.empty(shape=offspringSize)
            self.bestSol = np.ones((1, self.population.shape[1]))

          def calculateFitness(self):
            accs = np.zeros(shape=(self.population.shape[0],))
            idx = 0
            for currentSol in self.population:
                reducedFeaturesTrain = reduceFeature(currentSol, self.featuresTrain)
                reducedFeaturesTest = reduceFeature(currentSol, self.featuresTest)
                self.model.fit(reducedFeaturesTrain, self.labelsTrain)
                accs[idx] = self.model.score(reducedFeaturesTest, self.labelsTest)
                idx += 1
            return accs

          def selectMatingPool(self, fitness):
            for nParent in range(self.nosParentsMating):
                maxFitnessIdx = np.where(fitness == np.max(fitness))[0][0]
                self.parents[nParent, :] = self.population[maxFitnessIdx, :]
                fitness[maxFitnessIdx] = -99999999999

          # Single point crossover
          def crossover(self):
            crossoverPoint = np.uint8(self.offspring.shape[1] / 2)
            for k in range(self.offspring.shape[0]):
                parentIdx1 = k % self.parents.shape[0]
                parentIdx2 = (k + 1) % self.parents.shape[0]
                self.offspring[k, 0:crossoverPoint] = self.parents[parentIdx1, 0:crossoverPoint]
                self.offspring[k, crossoverPoint:] = self.parents[parentIdx2, crossoverPoint:]

          def mutation(self):
            mutationIdx = np.random.randint(low=0, high=self.offspring.shape[1], size=self.nosMutation)
            for i in range(self.offspring.shape[0]):
                self.offspring[i, mutationIdx] = 1 - self.offspring[i, mutationIdx]

          def logging(self, fitness, bestMatchIdx):
            bestSolIdx = np.where(self.bestSol == 1)[0]
            bestSolNumelements = bestSolIdx.shape[0]
            bestSolFitness = fitness[bestMatchIdx]
            print(f"Best Match Idx: {bestSolIdx}")
            print(f"Best Solution: {self.bestSol}")
            print(f"Number of Elements Selected: {bestSolNumelements}")
            print(f"Best solution fitness: {bestSolFitness}")
            plt.plot(fitness)
            plt.xlabel("Itrations")
            plt.ylabel("Fitness")
            plt.show()

          def saveLog(self, path):
            if not os.path.exists(path+"/"+ type(self.model).__name__):
              os.mkdir(path+"/"+ type(model).__name__)
            np.save(path+"/"+ type(self.model).__name__ + "/populationGA.npy", self.population)
            np.save(path+"/"+ type(self.model).__name__ + "/bestSolTillNowGA.npy", self.bestSol)
            return

          def gaMain(self, path):
            bestOutputs = []
            for gen in range(self.numGenerations):
                start = time()
                print(f"Generations {gen}/{self.numGenerations}")
                fitness = self.calculateFitness()
                bestOutputs.append(np.max(fitness))
                self.selectMatingPool(fitness)
                self.crossover()
                self.mutation()
                self.population[0:self.parents.shape[0], :] = self.parents
                self.population[self.parents.shape[0]:, :] = self.offspring
                end = time()
                bestMatchIdx = np.where(fitness == np.max(fitness))[0][0]
                self.bestSol = self.population[bestMatchIdx, :]
                self.saveLog(path)
                print(f"Time Taken {round((end-start),2)} secs Best Output "
                      f"{bestOutputs[-1]}")
            fitness = self.calculateFitness()
            bestMatchIdx = np.where(fitness == np.max(fitness))[0][0]
            self.bestSol = self.population[bestMatchIdx, :]
            self.logging(fitness, bestMatchIdx)
            return bestOutputs
