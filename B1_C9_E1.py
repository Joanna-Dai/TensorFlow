# predicting time series
import keras.metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define all functions

# attributes of TS
# trend: move in a specific direction
def trend(time, slope=0):
    return slope * time

# seasonality: repeating pattern over time
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,                # condition: season is before 0.4 of the year
                    np.cos(season_time * 2 * np.pi),  # if yes: cos(2*pi*x) (x: 0-->0.4, pattern(cos): 1-->-0.81) downward flattening
                    1 / np.exp(3 * season_time))      # if no: 1/exp(3*x)   (x: 0.4-->1, pattern(exp): 1.5 --> 2.7) upward flattening

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period # (remainder of time/period) / period, locating the season
    return amplitude * seasonal_pattern(season_time) # return seasonal pattern amplified by amplitude

# noise: seemingly random pertubations in a TS
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)  # generate random numbers in [0,1]
                                       # seed used to initialize the pseudo-random number generator. Can be any integer between 0 and 2**32 - 1 inclusive
    return rnd.randn(len(time)) * noise_level


# graph: plot the time series
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


# set parameters for trend, seasonality and noise
time = np.arange(4*365+1, dtype="float32") # 4-year time points starting from 0, numpy.arange(start=0, stop, step=1, dtype = None)
baseline = 10 # intercept
slope = 0.09 # trend
amplitude = 15 # amplify seasonality
noise_level = 6 # amplify noise
# create series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# update series with noise
series += noise(time, noise_level, seed=42)
# plot the time series
plot_series(time, series)


# split the data into training and validation sets
split_time = 1000
time_train = time[:split_time] # the first 1000 time points [0,1000)
x_train = series[:split_time] # returned series for training points
time_valid = time[split_time:] # the rest 461 time points starting from 1000, which is [1000, 1460] in this case
x_valid = series[split_time:] # returned series for validation points


# prediction 1 - naive forecast: predict series by taking the value at time t-1 to be the forecasted value at time t
naive_forecast = series[split_time-1: -1] # intentionally have 1 step difference
# show two series in the same graph
plt.figure(figsize=(10, 6))
plot = plot_series(time_valid, x_valid)
plot = plot_series(time_valid, naive_forecast)
plt.show()
# measure the prediction accuracy
# MSE: mean squared error
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy()) #76.475
# MAE: mean absolute error
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()) #6.899


# prediction 2 - moving average: average out a group of values and set that as the predicted value at t
def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)



