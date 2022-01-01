# using NASA weather data
# download time series weather csv data of SeaTac airport from https://data.giss.nasa.gov/cgi-bin/gistemp/stdata_show_v4.cgi?id=USW00024233&ds=14&dt=1

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# function to read weather data and create time series
def get_data():
    data_file = "C:/Users/user/tensorflow_datasets/station.csv"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    temperatures = []
    for line in lines:
        if line:
            linedata = line.split(',')
            linedata = linedata[1:13]
            for item in linedata:
                if item:
                    temperatures.append(float(item))  # temperatures will store all the temperature data as series/array

    # convert list to np.array: array is more efficient for large amount storage and numerical operation
    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype = 'float32')
    print(type(linedata))
    print(type(temperatures))
    print(type(series))
    return time, series

# call the function to load the data and create the time series
time, series = get_data()
print(time)
# normalize the time series
mean = series.mean(axis=0)
series-=mean #series-mean
std = series.std(axis=0)
series/=std #(series-mean)/std
print(type(series))

# split into training and validation sets
split_time = 792
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# windowed dataset
window_size = 24
batch_size = 12
shuffle_buffer_size = 48

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
valid_dataset = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

print(np.shape(dataset))

# simple RNN
model = tf.keras.models.Sequential([
    # RNN has an internal loop that iterates over the time steps of a sequence while maintaining an internal state of the time steps
    tf.keras.layers.SimpleRNN(100, return_sequences=True, input_shape=[None,1]),
    tf.keras.layers.SimpleRNN(100),
    # output layer
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(lr=1.5e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
history = model.fit(dataset, epochs=100, verbose=1, validation_data=valid_dataset)


# predict the values
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

forecast = model_forecast(model, series[..., np.newaxis], window_size)
results = forecast[split_time - window_size:-1]
print(results)
print(np.shape(results))

# plot the comparison
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

# measure prediction accuracy:
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
