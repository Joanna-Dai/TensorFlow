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
# model with keras tuner's hyperparameter tuning: train multi models. one with each possible set of parameters, evaluate the model to a metric you want
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

from kerastuner.tuners import RandomSearch

# hyperparameter parameter (hp) is used to control which values get changed
def build_model(hp):
  model = tf.keras.models.Sequential()
  # the layer will be tested with several input values, starting with 10 and increasing to 30 in steps of 2: i.e. train (30-10)/2=11 times (for layers)
  model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=10, max_value=30, step=2), activation='relu', input_shape=[window_size]))
  model.add(tf.keras.layers.Dense(10, activation='relu'))
  model.add(tf.keras.layers.Dense(1))

  # using hp.choice for a few options of 4 values for momentum (end with 4x11=44 possible combinations)
  model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(hp.Choice('momentum', values=[.9, .7, .5, .3]), lr=1e-5))
  return model

# randomsearch to manage all the iterations for this model:
#   objective is to minimize loss
#   cap the overall trials # to be max_trials
#   train and evaluate the model (eliminating random fluctuations) 3 times per trial
tuner = RandomSearch(build_model, objective='loss', max_trials=50, executions_per_trial=3, directory='my_dir', project_name='hello')

tuner.search_space_summary()

# start the search: train models with every possible hyperparameter according to your definition of the options to try
tuner.search(dataset, epochs=100, verbose=0)

# return top 10 trials based on the objective
# lowest score: 33.35; units: 26; momentum: 0.9
tuner.results_summary()

# get the bst 4 models: pick the one with units=28; momentum=0.5
models=tuner.get_best_models(num_models=4)
print(models)

# Based on tuner result: set neuron # of first layer = 28; momentum = 0.5 (also set lr=1e-5)
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    # neuron set to be 28
    tf.keras.layers.Dense(28, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# momentum set to be 0.5
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100,  verbose=1)

# put predicted value into forecast
forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

# measure prediction accuracy: MAE=4.48 (better than previous two)
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())