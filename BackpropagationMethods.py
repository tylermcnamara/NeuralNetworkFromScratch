import numpy
from numpy.ma.extras import average

from Network import Network
#Has the methods to train a network using backpropagation
class BackpropagationMethods:
    def __init__(self):
        self.deltas = None
        pass

    #Compute the deltas for the current sample
    def ComputeAllDeltas(self, net, input, y):
        N = len(net.Layers) - 1
        #Initialize matrix for delta values
        deltas = numpy.zeros(shape=(N + 1, N + 1))
        for i in range(0, net.Layers[N].NumNeurons):
            deltas[N, 0] += -1 * (y - net.Output(input)) * net.Layers[N].NeuronArray[i].APrime

        #Iterates for each layer counting down do to Backpropagation
        for layIndex in range(N, 0, -1):
            #Iterates for each neuron in Layer[layIndex]
            for neuIndex in range(0, net.Layers[layIndex].NumNeurons):
                #Iterates for each weight in current neuron, used to calculate delta for neurons in the prior layer
                for deltaIndex, weight in enumerate(net.Layers[layIndex].NeuronArray[neuIndex].Weights):
                    deltas[layIndex - 1, deltaIndex] += weight * deltas[layIndex, neuIndex] * net.Layers[layIndex - 1].NeuronArray[deltaIndex].APrime
        self.deltas = deltas

    #Compute the partial derivative of the error with respect to the weights based on the deltas
    def ComputeWeightPartials(self, net, input):
        allWeightPartials = [] #Initalize list to hold all partial derivatives of error wrt weights based on the deltas of neurons
        for k in range(0, len(net.Layers)): #Iterate through layers with k
            layerWeightPartials = numpy.zeros(shape=(net.Layers[k].NumNeurons, net.Layers[k].InputsPerNeuron))
            #Initialize numpy array to hold every weight in each neuron in the current layer
            for j in range(0, net.Layers[k].NumNeurons):
                for i in range(0, net.Layers[k].InputsPerNeuron):
                    #InputsPerNeuron is the number of weights connecting current neuron with neuron in layer k-1
                    if k != 0: #Weights from input to first hidden layer handled differently
                        layerWeightPartials[j, i] = self.deltas[k, j] *  net.Layers[k-1].NeuronArray[i].A
                    else:
                        layerWeightPartials[j, i] = self.deltas[k, j] * input[i, 0]
            allWeightPartials.append(layerWeightPartials)
        return allWeightPartials

    #Compute the partial derivative of the error with respect to the bias based on deltas
    def ComputerBiasPartials(self, net):
        allBiasPartials = [] #Initialize list to hold all partial derivatives of error wrt bias based on deltas of neurons
        for k in range(0, len(net.Layers)): #Iterate through layers with k
            layerBiasPartials = numpy.zeros(shape=net.Layers[k].NumNeurons,)
            #Initialize numpy array to hold the bias of each neuron in the current layer
            for j in range(0, net.Layers[k].NumNeurons):
                layerBiasPartials[j] = self.deltas[k, j] #Partials of error wrt bias is the delta of that neuron
            allBiasPartials.append(layerBiasPartials)
        return allBiasPartials

    #Update the weights of the network according to the partials and learning rate
    def UpdateWeightsGD(self, net, weightPartials, learningRate):
        for k in range(0, len(net.Layers)): #Iterate through layers with k
            currentWeights = net.Layers[k].GetWMatrix() #Get weights stored within current layer
            for j in range(0, net.Layers[k].NumNeurons): #Iterate through neurons in layer[k]
                for i in range(0, net.Layers[k].NeuronArray[j].WeightNum): # Iterate through the weights of neuron[j] in layer[k]
                    #Adjust weights based on learning rate and the partial of error wrt neuron[j]'s weights based on neuron[j]'s delta
                    currentWeights[j, i] = currentWeights[j, i] - learningRate * weightPartials[k][j][i]

            net.Layers[k].SetWMatrix(currentWeights)

    #Update the weights of the network according to the partials and learning rate
    def UpdateBiasesGD(self, net, biasPartials, learningRate):
        for k in range(0, len(net.Layers)): #Iterate through layers with k
            currentBiases = net.Layers[k].GetBMatrix() #Get weights stored within current layer
            for j in range(0, net.Layers[k].NumNeurons): #Iterate through each neuron in kth layer
                #Adjust weights based on learning rate and the partial of error wrt bias based on the neurons delta
                currentBiases[j, 0] = currentBiases[j, 0] - learningRate * biasPartials[k][j]
            net.Layers[k].SetBMatrix(currentBiases)



    #Iterative method for actual use in application
    def StochasticGradientDescent(self, net, inputs, targets, learningRate, numIterations):
        trainingSampleNum = len(inputs) #Determines how many samples are in the training set
        for currentIteration in range(0, numIterations): #Train for a fixed number of iterations
            for currentSample in range(0, trainingSampleNum): #Train for each sample
                currentInput = inputs[currentSample] #Get the current training input to the network
                currentTarget = targets[currentSample] #Get the target (ideal) output of the network
                output = net.Output(currentInput)
                #print(output)
                #Compute the deltas for the current sample
                self.ComputeAllDeltas(net, currentInput, currentTarget)
                #Compute the partial derivative of the error with respect to the weights based on the deltas
                weightPartials = self.ComputeWeightPartials(net, currentInput)
                #Compute the partial derivative of the error with respect to the bias based on deltas
                biasPartials = self.ComputerBiasPartials(net)
                #Update the weights of the network according to the partials and learning rate
                self.UpdateWeightsGD(net, weightPartials, learningRate)
                #Update the weights of the network according to the partials and learning rate
                self.UpdateBiasesGD(net, biasPartials, learningRate)
                #print("Input: ", currentInput, "Target: ", currentTarget, "Actual Output = ", net.Output(currentInput))

    #Generates an empty holder with the correct size numpy matrices for weights for batch GD to use
    def GenerateEmptyJWHolder(self, net):
        EmptyJWHolder = []
        for i in reversed(range(net.LayerNum)):
            matrixJWForLayer = numpy.zeros(shape=(net.Layers[i].NumNeurons, net.Layers[i].InputsPerNeuron))
            matrixJWForLayer = numpy.matrix.transpose(matrixJWForLayer)
            #Transpose so that it is in the same form as the matrix returned by ComputeJW
            EmptyJWHolder.append(matrixJWForLayer)
        return EmptyJWHolder

    #Generates and empty holder with the correct size numpy matrices for biases for batch GD to use
    def GenerateEmptyJBHolder(self, net):
        EmptyJBHolder = []
        for i in reversed(range(net.LayerNum)):
            matrixJBForLayer = numpy.zeros(shape=(net.Layers[i].NumNeurons, 1))
            EmptyJBHolder.append(matrixJBForLayer)
        return EmptyJBHolder

    def ApplyAverageToHolders(self, trainingSampleNum, holderJ):
        averageTerm = 1/trainingSampleNum
        for i in range(0, len(holderJ)):
            holderJ[i] = holderJ[i] * averageTerm

    def SumHolders(self, currentJW, holderJW):
        currentJW.reverse()
        for i in reversed(range(len(currentJW))):
            currentJW[i] = numpy.matrix.transpose(currentJW[i])
        for i in range(0, len(currentJW)):
            holderJW[i] = numpy.add(currentJW[i], holderJW[i])

    def UpdateWeightsBGD(self, net, weightHolder, learningRate):
        #Get weightHolder into the same order as the weights in each layer
        weightHolder.reverse()
        #Transpose each layer of weights in weightHolder because each neuron has more than one weight
        for i in range(0, len(weightHolder)):
            weightHolder[i] = numpy.matrix.transpose(weightHolder[i])
        #Continue now that weightHolder[k] and currentWeights of layer[k] have same shape
        for k in range(0, len(net.Layers)):  #Iterate through layers with k
            currentWeights = net.Layers[k].GetWMatrix()  #Get weights stored within current layer
            for j in range(0, net.Layers[k].NumNeurons):  #Iterate through neurons in layer[k]
                for i in range(0, net.Layers[k].NeuronArray[j].WeightNum):  # Iterate through the weights of neuron[j] in layer[k]
                    #Adjust weights based on learning rate and the partial of error wrt neuron[j]'s weights based on neuron[j]'s delta
                    currentWeights[j, i] = currentWeights[j, i] - learningRate * weightHolder[k][j][i]
            net.Layers[k].SetWMatrix(currentWeights)

    def UpdateBiasesBGD(self, net, biasHolder, learningRate):
        #Get biasHolder into same shape as layer[k]'s bias matrix
        biasHolder.reverse()
        for k in range(0, len(net.Layers)):  #Iterate through layers with k
            currentBiases = net.Layers[k].GetBMatrix()  #Get weights stored within current layer
            for j in range(0, net.Layers[k].NumNeurons):  #Iterate through each neuron in kth layer
                #Adjust weights based on learning rate and the partial of error wrt bias based on the neurons delta
                currentBiases[j, 0] = currentBiases[j, 0] - learningRate * biasHolder[k][0][j]
            net.Layers[k].SetBMatrix(currentBiases)


    def BatchGradientDescent(self, net, inputs, targets, learningRate, numIterations):
        trainingSampleNum = len(inputs) #Determine how many samples are in the training set
        for currentIteration in range(0, numIterations): #Train for a fixed number of iterations
            holderJW = self.GenerateEmptyJWHolder(net)
            holderJB = self.GenerateEmptyJBHolder(net)
            for currentSample in range(0, trainingSampleNum): #Train for each sample
                currentInput = inputs[currentSample] #Get the current training input to the network
                currentTarget = targets[currentSample] #Get the target (ideal) output of the network
                output = net.Output(currentInput)
                self.ComputeAllDeltas(net, currentInput, currentTarget)
                #Compute the weight partial derivatives for the current sample and store it using the SumHolders method
                weightPartials = self.ComputeWeightPartials(net, currentInput)
                self.SumHolders(weightPartials, holderJW)
                #Compute the bias partial derivatives for the current sample and store it using the SumHolders method
                biasPartials = self.ComputerBiasPartials(net)
                self.SumHolders(biasPartials, holderJB)
            self.ApplyAverageToHolders(trainingSampleNum, holderJW)
            self.ApplyAverageToHolders(trainingSampleNum, holderJB)
            #TODO Apply holderJW to update the weights of the network
            self.UpdateWeightsBGD(net, holderJW, learningRate)
            #TODO Apply holderJB to update the biases of the network
            self.UpdateBiasesBGD(net, holderJB, learningRate)

