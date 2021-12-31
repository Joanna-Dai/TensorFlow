# tunning hyperparameter for windowed time series

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# create time series dataset (as B1_C9_E1)
# define functions
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
# set parameters
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5
# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)
# split into training and validation sets
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# function to turn the time series into windowed dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # tf.data.Dataset.from_tensorflow_slices: turn a series into dataset
    dataset = tf.data.Dataset.from_tensor_slices(series)
    # split dataset into windows of window_size+1 items (the extra 1 will be used as label)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # split each window into features x (len=window_size) and label y (len=1)
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# set the parameters for windowed datasets
# split time series into training and validation series
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
# parameters for window
window_size = 20 # feature/input smaller sequences would be of 20 items
batch_size = 32 # there would be 32 batches of feature x and label y
shuffle_buffer_size = 1000

# training-ready dataset (tf.data.Dataset)
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# print one to inspect
for feature, label in dataset.take(1):
    print(feature) # 32 feature sets per batch (20 values in each feature set)
    print(label) # 32 labels per batch


# tf.data.Dataset can be passed to model.fit as a single parameter and tf.keras will take care of the rest
# simple DNN model:
model = tf.keras.models.Sequential([
    # input_shape: input data is features (each feature set is of 20 (window_size) items)
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    # output layer
    # that contains the 1-shape predicted value
    tf.keras.layers.Dense(1)
])
# loss function = mse, which is commonly used for regression problems
# hyperparameter tunning: learning rate (adjust from 1e-6 to callback: lr_schedule)
#   lrs callback: start the lr at 1e-8 and every epoch increase it by a small amount (bigger epoch until 100--> bigger lr until 1e-8*(10^5)=1e-3)
lr_scheldue = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
#   lr start with 1e-8
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
# use training dataset to train the model
# pre-adj 100-epoch result: loss=35.49
# add callbacks for lr
# post-adj 100-epoch
history = model.fit(dataset, epochs=100, callbacks=[lr_scheldue], verbose=1)

# plot loss against learning rate
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])

# model.predict, given series data, pass the model values from t to t+window_size and then get predicted value for the next time step
print(series[split_time:split_time+window_size]) #input series
# predicted next value (103.50): [np.newaxis] in model.predict() to keep the input shape consistent
print(model.predict(series[split_time:split_time+window_size][np.newaxis]))
# actual next value: 106.26
print(series[split_time+window_size])

# overall result of model.predict
forecast=[]
# time labels starting index of the input array (series) and the index value of predicted array (forecast)
for time in range(len(series)-window_size):
    forecast.append(
        model.predict(series[time:time+window_size][np.newaxis])
    )

# x_valid is from [split_time:]
# forecast's split_time is based on window_size points before split_time: is time label should start from split_time-window_size then
print(len(x_valid))
print(len(forecast)) #off by 20 (window_size)
print(forecast)
# rearrange the array to be in the same shape of x_valid
forecast = forecast[split_time-window_size:]
results = np.array(forecast)
print(results.shape)
# slice the array: for all elements, get (0,0) from the (461,1,1) array and make it (461,) array
results = results[:, 0, 0]
print(results.shape)
print(results)

# visualize the comparison between forecasted values vs. actual values
plt.figure(figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

# measure the accuracy: MAE=4.91
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())