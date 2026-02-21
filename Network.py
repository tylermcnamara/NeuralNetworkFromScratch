from Layer import Layer
import copy
import random

#The main.py neural network class
class Network:
    #main.py constructor
    def __init__(self, numInputs, numNeuronsPerLayer, actFunctionForLayer):
        self.LayerNum = len(numNeuronsPerLayer) #Number of layers
        self.Layers = [] #Variable to hold the layers
        self.Layers.append(Layer(numInputs, numNeuronsPerLayer[0], actFunctionForLayer[0]))
        #The first layer is set differently, fill in the rest of the layers
        for i in range(1, self.LayerNum):
            currentL = Layer(self.Layers[i-1].NumNeurons, numNeuronsPerLayer[i], actFunctionForLayer[i])
            self.Layers.append(currentL)
        self.RandomizeAll()

    def Output(self, input):
        currentOut = self.Layers[0].Output(input)
        for i in range(1, self.LayerNum):
            currentOut = self.Layers[i].Output(currentOut)
        return copy.deepcopy(currentOut)

    #Randomize the weights and biases of the network
    def RandomizeAll(self):
        #random.seed(0) #Only set a seed if needed for debugging
        for i in range(0, self.LayerNum):
            if i == (self.LayerNum - 1): #Last layer must be handled differently
                self.Layers[i].RandomizeLayerWeights(1, random)
                #each neuron in the output is
                self.Layers[i].RandomizeLayerBias(random)
            else:
                self.Layers[i].RandomizeLayerWeights(self.Layers[i+1].NumNeurons, random)
                #passes the neurons for fanout and the random number generator
                self.Layers[i].RandomizeLayerBias(random)

    def DisplayRand(self):
        for i in range(0, self.LayerNum):
            print("Layer: ", i)
            for j in range(0, self.Layers[i].NumNeurons):
                for k in range(0, self.Layers[i].NeuronArray[j].WeightNum):
                    print("Neuron: ", j, "Weight = ", self.Layers[i].NeuronArray[j].Weights[k])
                print("Neuron: ", j, "Bias: ", self.Layers[i].NeuronArray[j].Bias)

