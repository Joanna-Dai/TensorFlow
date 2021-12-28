# windowed time series dataset

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
    # output layer that contains the 1-shape predicted value
    tf.keras.layers.Dense(1)
])
# loss function = mse, which is commonly used for regression problems
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
# use training dataset to train the model
# pre-adj 100-epoch result: loss=35.49
model.fit(dataset, epochs=100, verbose=1)