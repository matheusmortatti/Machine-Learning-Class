import numpy as np
import cmath as math
import sys
import random

def ReLU(val):
    return max(0, val)
def dReLU(val):
    return 0 if val <= 0 else 1
def sigmoid(val):
    return 1 / (1 + math.exp(-val))
def dsigmoid(val):
    r = sigmoid(val)
    return r*(1 - r)

class NeuralNetworkModel:
    def __init__(self, l_hidden = 1, hidden_neurons = 16, activation = "ReLU", use_softmax = False, epochs = 10, batch_size = 1, alpha = 0.01):
        self.l_hidden = l_hidden
        self.hidden_neurons = hidden_neurons
        self.activation = np.vectorize(getattr(sys.modules[__name__],activation))
        self.dactivation = np.vectorize(getattr(sys.modules[__name__],"d" + activation))
        self.use_softmax = use_softmax
        self.epochs = epochs
        self.batch_size = batch_size

        self.alpha = alpha

        self.epsilon = 4
        

    def CreateNetwork(self, input, target):
        self.layers = []

        self.input = np.array(input)
        self.target = np.array(target)

        init_weight_func = np.vectorize(lambda i, j: self.randomize())

        input_layer = {}
        input_layer["a"] = np.zeros((self.input.shape[0], 1))
        input_layer["bias"] = np.zeros((self.hidden_neurons, 1))
        input_layer["weight"] = np.fromfunction(init_weight_func, (self.hidden_neurons, self.input.shape[0]))

        self.layers.append(input_layer)

        for i in range(self.l_hidden-1):
            new_l = {}
            new_l["a"] = np.zeros((self.hidden_neurons, 1))
            new_l["bias"] = np.ones((self.hidden_neurons, 1))
            new_l["weight"] = np.fromfunction(init_weight_func, (self.hidden_neurons, self.hidden_neurons))

            self.layers.append(new_l)
        
        self.class_number = np.amax(self.target)+1
        last_layer = {}
        last_layer["a"] = np.zeros((self.hidden_neurons, 1))
        last_layer["bias"] = np.ones((self.class_number, 1))
        last_layer["weight"] = np.fromfunction(init_weight_func, (self.class_number, self.hidden_neurons))

        self.layers.append(last_layer)
    
    def initialize_d(self):
        D = []
        D.append(np.zeros((self.hidden_neurons, self.input.shape[0])))

        for i in range(self.l_hidden-1):
            D.append(np.zeros((self.hidden_neurons, self.hidden_neurons)))
        
        D.append(np.zeros((self.class_number, self.hidden_neurons)))

        return D

    def randomize(self):
        return random.random() * (2 * self.epsilon) - self.epsilon

    def FeedForward(self, input):
        self.layers[0]["a"] = np.transpose(np.array([input]))

        for i in range(1, len(self.layers)):
            self.layers[i]["z"] = np.add(np.matmul(self.layers[i-1]["weight"], self.layers[i-1]["a"]), self.layers[i-1]["bias"])
            self.layers[i]["a"] = self.activation(self.layers[i]["z"])

        if self.use_softmax:
            return np.add(np.matmul(self.layers[-1]["weight"], self.layers[-1]["a"]), self.layers[-1]["bias"])
        else:
            return self.activation(np.add(np.matmul(self.layers[-1]["weight"], self.layers[-1]["a"]), self.layers[-1]["bias"]))
    
    def fit(self, input, target):
        self.CreateNetwork(input, target)
        
        nrow = self.input.shape[0]
        ncol = self.input.shape[1]

        it = 0
        divide_func = np.vectorize(lambda x: x / self.batch_size)
        while it < self.epochs:
            # shuffle input
        
            for offset in  range(0, ncol, self.batch_size):
                self.D = self.initialize_d()
                for col in range(0, self.batch_size):
                    sample = self.input[:,col]
                    starget = self.target[:,col]

                    output = self.FeedForward(sample)
                    self.BackPropagation(output, starget)
                
                # Take mean value
                for d in self.D:
                    divide_func(d)

                self.UpdateWeights()



            it += 1

    def UpdateWeights(self):
        for i in range(len(self.D)):
            self.layers[i]["weight"] = np.subtract(self.layers[i]["weight"], np.multiply(self.alpha, self.D[i]))
    
    def BackPropagation(self, output, target):
        output_error = np.subtract(output, target)

        self.D[-1] = np.add(self.D[-1], output_error)
        for i in range(len(self.layers)-1, 1, -1):
            self.D[i-1] = np.add(self.D[i-1],np.multiply(self.dactivation(self.layers[i]["z"]), np.matmul(np.transpose(self.layers[i]["weight"]), self.D[i])))

    def Predict(self, input, target):
        output = self.FeedForward(input)
        return output