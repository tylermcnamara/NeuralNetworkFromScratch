import numpy
import copy

#Neuron class
class Neuron:
    def __init__(self, numInputs, actFunction):
        self.Weights = numpy.empty(numInputs, dtype=float) #define the weights for the neuron
        self.WeightNum = len(self.Weights) #Variable to check the number of connections/weights the neuron has
        self.Bias = 0.0
        self.ActivationFunction = copy.deepcopy(actFunction)
        self.A = 0.0 #Current output A = f(z)
        self.APrime = 0.0 #Derivative of the activation function, APrime = f`(z)

    def Output(self, input):
        z = 0.0
        self.APrime = 0.0
        for i in range(0, self.WeightNum): #Sum the weighted outputs
            z += input[i] * self.Weights[i]
        z += self.Bias
        self.A = self.ActivationFunction.Output(z)
        self.APrime = self.ActivationFunction.OutputPrime(z)
        return self.A

    def RandomizeWeights(self, fanOut, randGenerator):
        r = 4 * numpy.sqrt(6/(self.WeightNum + fanOut)) #Get r
        for i in range(0, self.WeightNum): #Assign weights as random values between -r, r
            self.Weights[i] = randGenerator.uniform(-r, r)

    def RandomizeBias(self, randGenerator):
        self.Bias = randGenerator.uniform(0,1)







