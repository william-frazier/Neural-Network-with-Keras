"""
keras implementation of a neural network
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from diamond_data_reader import load_diamond_data

#random seed for reproducibility
np.random.seed(19)

#load data set
(X_train,Y_train,X_valid,Y_valid,X_test,Y_test) = load_diamond_data()

#create model
model = Sequential()
model.add(Dense(8, input_dim=8, 
                kernel_initializer='random_normal',
                bias_initializer='zeros',
                activation='tanh'))
model.add(Dense(120, kernel_initializer='random_normal',
                bias_initializer='zeros',
                activation='tanh'))
model.add(Dense(60, kernel_initializer='random_normal',
                bias_initializer='zeros',
                activation='tanh'))
model.add(Dense(1, kernel_initializer='random_normal',
                bias_initializer='zeros',
                activation='linear'))

#compile model
opt = SGD(lr=0.01)
model.compile(loss='mse',
                 optimizer=opt,
                 metrics = ['mse'])

#fit model
model.fit(X_train, Y_train, epochs=300, 
          batch_size=10, verbose=2,
          shuffle=True)

#evaluate the model
scores = model.evaluate(X_test, Y_test, batch_size = 10)
print(("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)))

#predict values
#print(model.predict(X_test, batch_size=10, verbose=0)*(18823-326)+326)
print(X_valid)
print(Y_valid)