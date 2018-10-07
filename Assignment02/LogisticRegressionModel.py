import numpy as np
import cmath as math
import sys
import os
import random
from tqdm import tqdm

def calculate_hfunction(features, thetas):
    h = sigmoid(np.matmul(thetas.T, features))
    return h
def sigmoid(val):
    return 1 / (1 + math.exp(-val))

class Model:
    """
    Class Constructor

    :param input: numpy 2D array. each collumn is a different data set example
    :param target: numpy 1D array. Each value is the correct label for the data set given
    :param epochs: integer. Number of epochs to use in training
    :param batch_size: integer. How many training examples to use to update the weights
    :param alpha: float. Learning Rate Param
    :param decay: float. By how quickly the network should decrease alpha param
    """
    def __init__(self, data, target, epochs = 1, batch_size = 1, alpha = 0.01, decay = 0.5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_classes = np.amax(target)+1
        self.decay = decay
        
        self.data = data

        n_examples = data.shape[1]
        self.target = np.zeros((self.n_classes, n_examples))

        for col in range(n_examples):
            self.target[target[col],col] = 1

        # Create one array of thetas for each possible class
        self.thetas_model = []
        for model in range (self.n_classes):
            self.thetas_model.append(np.random.uniform(-1, 1, self.data.shape[0]))
        
    """
    Perform Neural Network Training
    """
    def Fit(self):
        for model in range(self.n_classes):
            # tqdm.write('Training class ' + str(model+1) + ' / ' + str(self.n_classes))
            self.thetas_model[model] = self.logistic_regression(self.data, self.target[model,:], self.thetas_model[model], model)
            

    """
    Perform Neural Network Prediction

    :param data: numpy 2D array. each collumn is a different example to predict
    :param target: numpy 1D array. Each value is the correct label for the data set given
    """
    def Predict(self, data, target):
        sig = np.vectorize(sigmoid)
        cl = np.vectorize(lambda x: 1 if x.real > 0.5 else 0)

        n_examples = data.shape[1]
        target_pred = np.zeros((self.n_classes, n_examples))

        for col in range(n_examples):
            target_pred[target[col],col] = 1
        
        res = np.zeros(target.shape)
        classes = np.zeros(target.shape)
        for i in range(self.n_classes):
            
            t = self.thetas_model[i]
            m = target_pred[i,:]
            cand = (sig(np.matmul(data.T, t)))

            for k in tqdm(range(cand.shape[0]), desc="Predicting Class: " + str(i+1) + " / " + str(self.n_classes)):
                if res[k] < cand[k]:
                    res[k] = cand[k].real
                    classes[k] = i
        return classes

    
    """
    Perform Softmax Regression Training

    :param data: numpy 2D array. each collumn is a different example to predict
    :param target: numpy 1D array. Each value is the correct label for the data set given
    :param theta: numpy 2D array. Each collumn is a different set of weights
    """
    def logistic_regression(self, data, target, thetas, n_class):
        m          = data.shape[1]
        iterations = 0

        #startTime = current_time()
        nrow = data.shape[0]
        ncol = data.shape[1]
        
        while(iterations < self.epochs):

            # Step through the dataset in chuncks
            for col in tqdm(range(0, ncol, self.batch_size), desc="Class: " + str(n_class) + " / " + str(self.n_classes) + ' Epochs: ' + str(iterations+1) + ' / ' + str(self.epochs)):

                s = np.zeros((nrow))
                # We add every row of the dataset to the error calculation (Batch)
                for offset in range(self.batch_size):
                    if col + offset >= m:
                        break
                    
                    sample = data[:,offset + col]
                    starget = target[offset + col]

                    h = calculate_hfunction(sample, thetas)
                    s = s + (h - starget) * sample

                # Updating the new thetas vector values
                thetas = thetas - ((self.alpha / self.batch_size) * s)
                
            iterations = iterations + 1
            self.alpha *= 1/(1 + self.decay*iterations)
        
        return thetas