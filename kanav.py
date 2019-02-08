import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


#NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 10,init = 'uniform',activation = 'relu',input_dim = 784) )
#classifier.add(Dense(output_dim = 20,init = 'uniform',activation = 'relu') )
classifier.add(Dense(output_dim = 10,init = 'uniform',activation = 'relu') )
classifier.add(Dense(output_dim = 10,init = 'uniform',activation = 'softmax') ) #Activation Method would be Sigmoid here

classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
classifier.fit(X,y,batch_size = 10,nb_epoch = 100)

y_pred = classifier.predict(Xtest)


for i in range(28000):
    y_pred[i] = y_pred[i] == y_pred[i].max()

result = []
for i in y_pred:
    if np.all(i == np.array([1,0,0,0,0,0,0,0,0,0],dtype = np.float32)):
        result.append(0)
    elif np.all(i == np.array([0,1,0,0,0,0,0,0,0,0],dtype = np.float32)):
        result.append(1)
    elif np.all(i == np.array([0,0,1,0,0,0,0,0,0,0],dtype = np.float32)):
        result.append(2)
    elif np.all(i == np.array([0,0,0,1,0,0,0,0,0,0],dtype = np.float32)):
        result.append(3)
    elif np.all(i == np.array([0,0,0,0,1,0,0,0,0,0],dtype = np.float32)):
        result.append(4)
    elif np.all(i == np.array([0,0,0,0,0,1,0,0,0,0],dtype = np.float32)):
        result.append(5)
    elif np.all(i == np.array([0,0,0,0,0,0,1,0,0,0],dtype = np.float32)):
        result.append(6)
    elif np.all(i == np.array([0,0,0,0,0,0,0,1,0,0],dtype = np.float32)):
        result.append(7)
    elif np.all(i == np.array([0,0,0,0,0,0,0,0,1,0],dtype = np.float32)):
        result.append(8)
    elif np.all(i == np.array([0,0,0,0,0,0,0,0,0,1],dtype = np.float32)):
        result.append(9)
    

df = pd.DataFrame(result, columns=["Answer"])
df.to_csv('NeuralNetwork.csv', index=False)
"""

        """