import numpy as np
import cmath as math
import sys
import os
import random
from tqdm import tqdm

def ReLU(val):
    return max(0, val)
def dReLU(val):
    return 0 if val <= 0 else 1
def sigmoid(val):
    return 1 / (1 + math.exp(-val))
def dsigmoid(val):
    r = sigmoid(val)
    return r*(1 - r)

class Model:
    def __init__(self, input, target, l_hidden = 1, hidden_neurons = 16, activation = "sigmoid", use_softmax = False, epochs = 10, batch_size = 1, alpha = 0.01):
        self.l_hidden = l_hidden
        self.hidden_neurons = hidden_neurons
        self.activation = np.vectorize(getattr(sys.modules[__name__],activation))
        self.dactivation = np.vectorize(getattr(sys.modules[__name__],"d" + activation))
        self.use_softmax = use_softmax
        self.epochs = epochs
        self.batch_size = batch_size
        self.v_exp = np.vectorize(math.exp)

        self.alpha = alpha

        self.epsilon = 1

        self.CreateNetwork(input, target)
        

    def CreateNetwork(self, input, target):
        self.layers = []

        self.input = np.array(input)
        self.target = np.array(target)

        init_weight_func = np.vectorize(lambda i, j: self.randomize())

        input_layer = {}
        input_layer["a"] = np.zeros((self.input.shape[0], 1))
        input_layer["bias"] = np.fromfunction(init_weight_func, (self.hidden_neurons, 1))
        input_layer["weight"] = np.fromfunction(init_weight_func, (self.hidden_neurons, self.input.shape[0]))

        self.layers.append(input_layer)

        for i in range(self.l_hidden-1):
            new_l = {}
            new_l["a"] = np.zeros((self.hidden_neurons, 1))
            new_l["bias"] = np.fromfunction(init_weight_func, (self.hidden_neurons, 1))
            new_l["weight"] = np.fromfunction(init_weight_func, (self.hidden_neurons, self.hidden_neurons))

            self.layers.append(new_l)
        
        self.class_number = np.amax(self.target)+1
        last_layer = {}
        last_layer["a"] = np.zeros((self.hidden_neurons, 1))
        last_layer["bias"] = np.fromfunction(init_weight_func, (self.class_number, 1))
        last_layer["weight"] = np.fromfunction(init_weight_func, (self.class_number, self.hidden_neurons))

        self.layers.append(last_layer)
    
    def initialize_error(self):
        error = []

        error.append(np.zeros((self.input.shape[0], 1)))

        for i in range(self.l_hidden):
            error.append(np.zeros((self.hidden_neurons, 1)))
        
        error.append(np.zeros((self.class_number,1)))

        return error
    
    def initialize_d(self):
        D = []
        bD = []

        D.append(np.zeros((self.hidden_neurons, self.input.shape[0])))
        bD.append(np.zeros((self.hidden_neurons, 1)))

        for i in range(self.l_hidden-1):
            D.append(np.zeros((self.hidden_neurons, self.hidden_neurons)))
            bD.append(np.zeros((self.hidden_neurons, 1)))
        
        D.append(np.zeros((self.class_number, self.hidden_neurons)))
        bD.append(np.zeros((self.class_number, 1)))

        return D, bD

    def randomize(self):
        return random.random() * (2 * self.epsilon) - self.epsilon

    def FeedForward(self, input):
        self.layers[0]["a"] = np.transpose(np.array([input]))

        for i in range(1, len(self.layers)):
            self.layers[i]["z"] = np.add(np.matmul(self.layers[i-1]["weight"], self.layers[i-1]["a"]), self.layers[i-1]["bias"])
            self.layers[i]["a"] = self.activation(self.layers[i]["z"])
            # print(self.layers[i]["a"])

        output_layer = np.add(np.matmul(self.layers[-1]["weight"], self.layers[-1]["a"]), self.layers[-1]["bias"])
        if self.use_softmax:
            e = self.v_exp(output_layer)
            es = np.sum(e)
            div = np.vectorize(lambda x: x / es)
            return div(e)
        else:
            return self.activation(output_layer)
    

    """
    Perform Neural Network Training

    :param input: numpy 2D array. each collumn is a different training example
    :param target: numpy 1D array. Each value is the correct label for the training example
    """
    def fit(self):
        
        nrow = self.input.shape[0]
        ncol = self.input.shape[1]

        if self.batch_size > ncol:
            self.batch_size = ncol

        it = 0
        divide_func = np.vectorize(lambda x: x / self.batch_size)
        while it < self.epochs:
            # shuffle input
            
            rng_state = np.random.get_state()
            np.random.shuffle(self.input.T)
            np.random.set_state(rng_state)
            np.random.shuffle(self.target.T)

            for offset in tqdm(range(0, ncol, self.batch_size), desc='Epochs: ' + str(it+1) + ' / ' + str(self.epochs)):
                self.D, self.bD = self.initialize_d()
                for col in range(self.batch_size):
                    if(offset + col >= ncol):
                        break
                    
                    self.error = self.initialize_error()

                    sample = self.input[:,offset + col]
                    starget = np.zeros((self.class_number, 1))
                    target_class = self.target[offset + col]
                    starget[target_class,0] = 1

                    output = self.FeedForward(sample)
                    self.BackPropagation(output, starget)
                
                # Take mean value
                for d in self.D:
                    d = divide_func(d)
                for d in self.bD:
                    d = divide_func(d)

                self.UpdateWeights()

            it += 1

    def UpdateWeights(self):
        for i in range(len(self.D)):
            self.layers[i]["weight"] -= np.multiply(self.alpha, self.D[i])
            self.layers[i]["bias"] -= np.multiply(self.alpha, self.bD[i])
    
    def BackPropagation(self, output, target):
        output_error = np.subtract(output, target)
        self.error[-1] = np.add(self.error[-1], output_error)

        for i in range(len(self.layers)-1, 0, -1):
            txd = np.matmul(np.transpose(self.layers[i]["weight"]),self.error[i+1])
            txdxg = np.multiply(self.dactivation(self.layers[i]["z"]), txd)
            self.error[i] = np.add(self.error[i],txdxg)
        
        for k in range(len(self.layers)):
            a = self.layers[k]["a"]
            err = self.error[k+1]

            for i in range(err.shape[0]):
                self.D[k][i,:] = (a*err[i])[:,0]
                self.bD[k][i] = err[i]


    def Predict(self, data, target):
        y_pred = np.zeros((len(target)))
        for i in range(len(target)):
            res = {}
            res["output"] = self.FeedForward(data[:,i])
            res["predicted_class"] = np.argmax(res["output"])

            y_pred[i] = res["predicted_class"]

        return y_pred