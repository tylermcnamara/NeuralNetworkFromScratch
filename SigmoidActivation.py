import math
from AbstractActivationFunction import AbstractActivationFunction

class SigmoidActivation(AbstractActivationFunction):
    def __init__(self): #Empty constructor since the activation function doesn't need anything
        pass

    def Output(self, x):
        y = 1.0/(1.0 + math.exp(-x)) #1/(1-e^-x)
        return y

    def OutputPrime(self, x):
        y = self.Output(x) * (1-self.Output(x)) # f(x) * (1 - f(x)) = f'(x)
        return y