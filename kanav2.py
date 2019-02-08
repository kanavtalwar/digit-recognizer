import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:785].values
y = dataset.iloc[:,0].values
y = y.reshape(42000,1)
testset = pd.read_csv('test.csv')
Xtest = testset.iloc[:, :].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000,criterion = 'gini',random_state = 0)
classifier.fit(X,y) 
# Predicting the Test set results
y_pred = classifier.predict(Xtest)

"""
for i in range(28000):
    y_pred[i] = y_pred[i] == y_pred[i].max()
"""
neuralset = pd.read_csv('NeuralNetwork.csv')
ytemp = neuralset.iloc[:,1].values
result = []
count = 0
for i in y_pred:
    if np.all(i == np.array([1,0,0,0,0,0,0,0,0,0],dtype = np.float64)):
        result.append(0)
    elif np.all(i == np.array([0,1,0,0,0,0,0,0,0,0],dtype = np.float64)):
        result.append(1)
    elif np.all(i == np.array([0,0,1,0,0,0,0,0,0,0],dtype = np.float64)):
        result.append(2)
    elif np.all(i == np.array([0,0,0,1,0,0,0,0,0,0],dtype = np.float64)):
        result.append(3)
    elif np.all(i == np.array([0,0,0,0,1,0,0,0,0,0],dtype = np.float64)):
        result.append(4)
    elif np.all(i == np.array([0,0,0,0,0,1,0,0,0,0],dtype = np.float64)):
        result.append(5)
    elif np.all(i == np.array([0,0,0,0,0,0,1,0,0,0],dtype = np.float64)):
        result.append(6)
    elif np.all(i == np.array([0,0,0,0,0,0,0,1,0,0],dtype = np.float64)):
        result.append(7)
    elif np.all(i == np.array([0,0,0,0,0,0,0,0,1,0],dtype = np.float64)):
        result.append(8)
    elif np.all(i == np.array([0,0,0,0,0,0,0,0,0,1],dtype = np.float64)):
        result.append(9)
    else:
        #result.append(randint(0,9))
        result.append(ytemp[count])
    count = count + 1        
       
    

df = pd.DataFrame(result, columns=["Label"])
df.to_csv('Random Forest.csv', index=False)