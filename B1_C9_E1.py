# predicting time series
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# plot time series (TS): x is temporal by nature (single y: univariate; multiple y: multivariate)
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

# attribute of TS: move in a specific direction
def trend(time, slope=0):
    return slope * time

# seasonality: repeating pattern over time
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

# noise: seemingly random pertubations in a TS
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level