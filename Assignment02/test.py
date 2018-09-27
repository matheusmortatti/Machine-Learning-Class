import NeuralNetworkModel as NNM
import numpy as np
from sklearn.datasets import load_digits

data, target = load_digits(10, True)
data = np.transpose(data)

half = len(target)//2

data_train = data[:,:half]
target_train = target[:half]
data_val = data[:,half:]
target_val = target[half:]

model = NNM.Model(activation="ReLU", epochs=10000, alpha=0.00001, l_hidden=1, batch_size=1, use_softmax=True)
print("training ...")
model.fit(data_train, [target_train])
print("training done")

correct = 0
for i in range(len(target_val)):
    res = model.Predict(data_val[:,i],target_val[i])

    if res["predicted_class"] == target_val[i]:
        correct += 1
    
print("accurracy = ", correct / len(target_val))