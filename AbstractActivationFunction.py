#This is the abstract class that all activation functions should inherit from
from abc import ABCMeta, abstractmethod

class AbstractActivationFunction(metaclass=ABCMeta):
    def Output(self, x):
        # Output the activation function from input x
        raise NotImplementedError("The activation function you are using does not implement an Output method.")

    def OutputPrime(self, x):
        #Output the derivative of the activation function from input x
        raise NotImplementedError("The activation function you are using does not implement an OutputPrime method.")