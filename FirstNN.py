import numpy
from Network import Network
from BackpropagationMethods import BackpropagationMethods
from SigmoidActivation import SigmoidActivation

inputS1=numpy.zeros(shape=(2,1)) #Allocate memory for the input
inputS1[0,0]=0.3 #Store the value of x0
inputS1[1,0]=0.5 #Store the value of x1
outputS1=0.1
#Second Sample
inputS2=numpy.zeros(shape=(2,1)) #Allocate memory for the input
inputS2[0,0]=0.5 #Store the value of x0
inputS2[1,0]=0.9 #Store the value of x1
outputS2=0.2
#Store the input and output for each sample in an array
input=[] #This is the array that holds all the training samples input data
input.append(inputS1)
input.append(inputS2)

output=[] #This is the array that holds all the training samples output(target) data
output.append(outputS1)
output.append(outputS2)
#Network and associated parameters
learningRate = 0.5
numNeuronsPerLayer = [3, 2, 1]
actFunctions = [SigmoidActivation(), SigmoidActivation(), SigmoidActivation()]
n1 = Network(2, numNeuronsPerLayer, actFunctions)
#Set dummy weighting for testing
#first layer
n1.Layers[0].NeuronArray[0].Weights[0] = 0.2
n1.Layers[0].NeuronArray[0].Weights[1] = 0.3
n1.Layers[0].NeuronArray[0].Bias = 0.0
n1.Layers[0].NeuronArray[1].Weights[0] = 0.1
n1.Layers[0].NeuronArray[1].Weights[1] = 0.25
n1.Layers[0].NeuronArray[1].Bias = 0.0
n1.Layers[0].NeuronArray[2].Weights[0] = 0.5
n1.Layers[0].NeuronArray[2].Weights[1] = 0.6
n1.Layers[0].NeuronArray[2].Bias = 0.0
#second layer
n1.Layers[1].NeuronArray[0].Weights[0] = 0.8
n1.Layers[1].NeuronArray[0].Weights[1] = 0.1
n1.Layers[1].NeuronArray[0].Weights[2] = 0.7
n1.Layers[1].NeuronArray[0].Bias = 0.0
n1.Layers[1].NeuronArray[1].Weights[0] = 0.2
n1.Layers[1].NeuronArray[1].Weights[1] = 0.1
n1.Layers[1].NeuronArray[1].Weights[2] = 0.3
n1.Layers[1].NeuronArray[1].Bias = 0.0
#output layer
n1.Layers[2].NeuronArray[0].Weights[0] = 0.5
n1.Layers[2].NeuronArray[0].Weights[1] = 0.3
n1.Layers[2].NeuronArray[0].Bias = 0.0

BackpropagationMethods = BackpropagationMethods()




#BackpropagationMethods.StochasticGradientDescent(n1, input, output, learningRate, 1)
#print(n1.Output(inputS2))
BackpropagationMethods.BatchGradientDescent(n1, input, output, learningRate, 1)
print(n1.Output(inputS2))


