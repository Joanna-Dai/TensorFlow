
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#dense: densely connected neurons (every neuron is connected to neuron of next layer)
#units=1: one dense layer with one neuron in our entire neural network
#input_shape=1: shape of input data (X) is single value
l0 = Dense(units=1, input_shape=[1])

#sequntial: define layers
model = Sequential([l0])

#model.compile: model approach
#optimizer='sgd':stochastic gradient descent, the way to guess and calcuate loss
#loss='mean_squared_error': measure for loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

#input data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#fit the Xs to the Ys and try it 500 times
model.fit(xs, ys, epochs=500)

#predict the output based X=10
print(model.predict([10.0]))

#the model only has a single neuro and that neuron learns a weight(W) and bias(B) Y=WX + B
print("Here is what I learned: {}".format(l0.get_weights()))

