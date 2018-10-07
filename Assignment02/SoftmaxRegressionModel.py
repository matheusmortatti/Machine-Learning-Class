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
def calculate_cost_function(thetas, data, target):
    m = data.shape[1]

    res = np.matmul(data.T, thetas)
    
    s = np.sum((res-target)*(res-target))
    
    return (1/(2*m)) * (s)

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
    def __init__(self, data, target, epochs = 10, batch_size = 1, alpha = 0.01, decay = 0.5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_classes = np.amax(target)+1

        self.decay = decay

        self.v_exp = np.vectorize(math.exp)
        
        self.data = data

        n_examples = data.shape[1]
        self.target = target

        # Create one array of thetas for each possible class
        self.thetas_model = np.random.uniform(-1, 1, (self.data.shape[0],self.n_classes))
        
    """
    Perform Neural Network Training
    """
    def Fit(self):
        self.thetas_model = self.softmax_regression(self.data, self.target, self.thetas_model)
            
    """
    Perform Neural Network Prediction

    :param data: numpy 2D array. each collumn is a different example to predict
    :param target: numpy 1D array. Each value is the correct label for the data set given
    """
    def Predict(self, data, target):
        sig = np.vectorize(sigmoid)
        
        y_pred = np.zeros((len(target)))
        for i in tqdm(range(data.shape[1]), desc="Predicting"):
            sample = data[:,i]
            

            res = sig(np.matmul(sample.T, self.thetas_model))
            cl = np.argmax(res)

            y_pred[i] = cl

        return y_pred

    """
    Perform Softmax Regression Training

    :param data: numpy 2D array. each collumn is a different example to predict
    :param target: numpy 1D array. Each value is the correct label for the data set given
    :param theta: numpy 2D array. Each collumn is a different set of weights
    """
    def softmax_regression(self, data, target, thetas):
        m          = data.shape[1]
        iterations = 0

        # After j_step iterations, compute cost function
        costs       = []
        itr_numbers = []
        
        retryCount = 0
        retryMax = 1000

        #startTime = current_time()
        nrow = data.shape[0]
        ncol = data.shape[1]
        
        while(iterations < self.epochs):

            # Step through the dataset in chuncks
            for col in tqdm(range(0, ncol, self.batch_size), desc='Epochs: ' + str(iterations+1) + ' / ' + str(self.epochs)):

                s = np.zeros((nrow, self.n_classes))
                # We add every row of the dataset to the error calculation (Batch)
                for offset in range(self.batch_size):
                    if col + offset >= m:
                        break
                    
                    sample = data[:,offset + col]
                    starget = np.zeros((self.n_classes, 1))
                    target_class = self.target[offset + col]
                    starget[target_class,0] = 1

                    e = self.v_exp(np.matmul(thetas.T,sample))
                    es = np.sum(e)
                    div = np.vectorize(lambda x: x / es)

                    h = div(e)[np.newaxis].T
                    s = s + np.matmul((h - starget),sample[np.newaxis]).T

                # Updating the new thetas vector values
                thetas = thetas - ((self.alpha / self.batch_size) * s)
                
            iterations = iterations + 1
            self.alpha *= 1/(1 + self.decay*iterations)
        
        return thetas