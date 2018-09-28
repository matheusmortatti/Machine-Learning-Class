import NeuralNetworkModel as NNM
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

dataset_train, y_true = load_digits(10, True)
dataset_train = np.transpose(dataset_train)

half = len(y_true)//2

data_train = dataset_train[:,:half]
target_train = y_true[:half]
data_val = dataset_train[:,half:]
target_val = y_true[half:]

model = NNM.Model(activation="sigmoid", epochs=1, alpha=0.01, l_hidden=1, hidden_neurons=128, batch_size=1, use_softmax=False)
model.CreateNetwork(data_train, [target_train])

acc = [0]
for i in range(50):
    print("training ", i, '/10')
    model.fit(data_train, [target_train])
    print("training done")

    correct = 0
    y_pred = np.zeros((len(target_val)))
    for i in range(len(target_val)):
        res = model.Predict(data_val[:,i],target_val[i])

        y_pred[i] = res["predicted_class"]
        if res["predicted_class"] == target_val[i]:
            correct += 1

    accuracy = correct / len(target_val)
    if accuracy < acc[-1]:
        model.alpha /= 2

    acc.append(accuracy)
    print("accurracy = ", accuracy)
    print(confusion_matrix(target_val, y_pred))
    print(classification_report(target_val, y_pred))

print(acc)