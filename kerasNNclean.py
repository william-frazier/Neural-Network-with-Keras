"""
keras implementation of a neural network
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np
np.set_printoptions(threshold=np.nan)
#to print full numpy arrays

import matplotlib.pyplot as plt

from diamond_data_reader import load_diamond_data

#random seed for reproducibility
np.random.seed(19)

#load data set
(X_train,Y_train,X_valid,Y_valid,X_test,Y_test) = load_diamond_data()

#run main function
def run_model():
    """
    Runs the neural network training and prediction functions, prompting
    the user for the number of epochs to train over and the size of the
    slice of the validation or test sets to use for predictions.
    """
    num_epochs = int(input("Enter the number of epochs to run: "))
    size_pred = int(input("Enter the size of the prediction slice: "))
    model = create_model()
    compile_model(model)
    history = fit_model(model, num_epochs)
    plot_loss(history)
    evaluate_model(model)
    predict_prices(model,size_pred)

#create model
def create_model():
    """
    Initializes the sequential neural network architecture with an input
    layer, 3 hidden layers, and output layer. The activation function is
    tanh except for the output layer which uses a linear function. The
    parameter weights are initilized with a random normal distribution.
    """
    model = Sequential()
    model.add(Dense(8, input_dim=8, 
                    kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(256, kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(128, kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(64, kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(8, kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='linear'))
    return model

def compile_model(model):
    """
    Compiles the neural network model using stochastic gradient descent
    as an optimizer and mean squared error as the loss function.
    """
    opt = SGD(lr=0.05)
    model.compile(loss='mse',
                     optimizer=opt,
                     metrics = ['mse'])

def fit_model(model,num_epochs):
    """
    Fits the model to the training set using batches over a given number
    of epochs. The batch is randomly chosen from the training set.
    """
    history = model.fit(X_train, Y_train, epochs=num_epochs, 
              batch_size=4, verbose=2,
              shuffle=True)
    return history

def plot_loss(model_history):
    """
    Plots the mean squared error as a function of the number of epochs.
    """
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def evaluate_model(model):
    """
    Prints the mean squared error using the validation sets.
    """
    scores = model.evaluate(X_test, Y_test, batch_size = 4)
    scoresv = model.evaluate(X_valid, Y_valid, batch_size = 4)
    print(("\n%s on test set: %.12f%%" % (model.metrics_names[1], scores[1]*100)))
    print(("%s on validation set: %.12f%%" % (model.metrics_names[1], scoresv[1]*100)))

def price_differences(actual_prices,predictions,index_val,percent):
    """
    Finds the elementwise difference in prices between a slice of the predictions
    and a slice of the validation set, rounding to the nearest 2 decimal places. If
    the parameter 'percent' == True, will instead return the elementwise percent error
    between the slices.
    """
    size_predictions = len(predictions[:index_val])
    size_actual = len(actual_prices[:index_val])
    diff = abs((predictions*(18823-326)+326) - (actual_prices*(18823-326)+326))
    dn_predictions = np.reshape(predictions[0:index_val], (size_predictions, 1))*(18823-326)+326
    dn_actual = np.reshape(actual_prices[0:index_val], (size_actual,1))*(18823-326)+326
    raw_diff =np.squeeze(dn_predictions - dn_actual)
    #per_diff = np.divide(np.reshape(diff, (len(predictions),1)),actual_prices)*100
    percent_diff = np.divide(np.reshape(raw_diff, (len(dn_predictions),1)),dn_actual)*100
    if percent:
        return np.around(percent_diff,2)
    else:
        print("The minimum difference =", np.around(min(diff),2))
        print("The maximum difference =", np.around(max(diff),2))
        print("The average difference =", np.around(np.mean(diff),2))
        #print("The average difference by % =", np.around(np.mean(per_diff),2))
        return np.squeeze(np.around(raw_diff,2))

def predict_prices(model,size_pred):
    """
    Makes predictions using the fitted model on a given slice of the validation set.
    Prints max and min predicted prices and actual prices as a comparison metric and
    prints elementwise differences as described in price_differences().
    """
    predictions = np.squeeze(model.predict(X_test, batch_size=32, verbose=0))
    print("Max predicted price is: " + str(max(predictions*(18823-326)+326)))
    print("Max actual price is: " + str(max(Y_test)*(18823-326)+326))
    print("Some price differences by percent (divide by 100 to get actual percent) are: " + str(price_differences(Y_test,predictions,size_pred,True)))
    print("Min predicted price is: " + str(min(predictions*(18823-326)+326)))
    print("Min actual price is: " + str(min(Y_test)*(18823-326)+326))
    print("Some price differences by value are: " + str(price_differences(Y_test,predictions,size_pred,False)))
