import numpy as np
import cmath as math
import sys
import os
import random
import progressbar

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
    def __init__(self, data, target, epochs = 10, batch_size = 1, alpha = 0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_classes = np.amax(target)+1
        
        self.data = data

        n_examples = data.shape[1]
        self.target = np.zeros((self.n_classes, n_examples))

        for col in range(n_examples):
            self.target[target[col],col] = 1

        # Create one array of thetas for each possible class
        self.thetas_model = []
        for model in range (self.n_classes):
            self.thetas_model.append(np.random.uniform(-1, 1, self.data.shape[0]))
        
    def Fit(self):
        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=self.n_classes).start()
        for model in range(self.n_classes):
            bar.update(model)
            self.thetas_model[model] = self.logistic_regression(self.data, self.target[model,:], self.thetas_model[model])
        bar.finish()
            

    def Predict(self, data, target):
        sig = np.vectorize(sigmoid)
        cl = np.vectorize(lambda x: 1 if x.real > 0.5 else 0)

        n_examples = data.shape[1]
        target_pred = np.zeros((self.n_classes, n_examples))

        for col in range(n_examples):
            target_pred[target[col],col] = 1
        
        acc = 0
        for i in range(self.n_classes):
            
            t = self.thetas_model[i]
            m = target_pred[i,:]
            res = cl(sig(np.matmul(data.T, t)))

            for k in range(res.shape[0]):
                if m[k] == res[k]:
                    acc+=1
        print("Accuracy:",acc / (self.n_classes * res.shape[0]))

    def logistic_regression(self, data, target, thetas, j_step=1):
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
            for col in range(0, ncol, self.batch_size):

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
                
            # keep a new cost value
            if iterations % j_step == 0:
                cost = calculate_cost_function(thetas, data, target)
                if len(costs)>0 and cost > costs[-1]:
                    self.alpha /= 1.001
                    if retryCount < retryMax:
                        retryCount += 1
                    else:
                        iterations = max_iterations
                else:
                    retryCount = 0
                costs.append(cost)
                itr_numbers.append(iterations)
                
            iterations = iterations + 1
        
        return thetas