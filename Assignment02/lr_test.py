import LogisticRegressionModel as LRM
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas

# Read and treat training dataset
dataset_train = pandas.read_csv('fashion-mnist-dataset/fashion-mnist_train.csv').values #np.genfromtxt('fashion-mnist-dataset/fashion-mnist_train.csv', delimiter=',')
y_true = dataset_train[:,0]
dataset_train = np.delete(dataset_train, 0, 1).T
dataset_train = dataset_train / dataset_train.max()

# Read and treat test dataset
dataset_test = pandas.read_csv('fashion-mnist-dataset/fashion-mnist_test.csv').values
target_test = dataset_test[:,0]
dataset_test = np.delete(dataset_test, 0, 1).T
dataset_test = dataset_test / dataset_test.max()

# dataset_train, y_true = load_digits(10, True)
# dataset_train = np.transpose(dataset_train)

half = len(y_true)//2

data_train = dataset_train[:,:half]
target_train = y_true[:half]
data_val = dataset_train[:,half:]
target_val = y_true[half:]

model = LRM.Model(data_train, target_train, epochs=10, alpha=0.01, batch_size=1)
model.Fit()
y_pred = model.Predict(data_val, target_val)

print(confusion_matrix(target_val, y_pred))
print(classification_report(target_val, y_pred))