# create windowed dataset: structure the time series data for training predictive model
#   to turn any time series into a set of training data for a neural network
import tensorflow as tf

# create basic dataset containing 0-9
dataset = tf.data.Dataset.range(10)
# dataset.window(5): split dataset into windows of 5 items
# shift=1: each window shifts one spot from the previous one
# drop_remainder=True: once it reached the point close to the end of dataset where the windows would be smaller than size=5, they should be dropped
dataset = dataset.window(5, shift=1, drop_remainder=True)
# dataset.flat_map: the process of splitting the dataset (after defining the window)
dataset = dataset.flat_map(lambda window: window.batch(5))
# print the windows in the dataset
for window in dataset:
  print(window.numpy())
# split each window into features x (everything before the last value) and label y (the last value)
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
# print features x and label y
for x,y in dataset:
  print(x.numpy(), y.numpy())
# shuffle the data in dataset: source data set --> random buffer --> a data set will be taken from the buffer
# the buffer_size will impact the randomness: buffer_size>=dataset size-->uniform shuffle; buffer_size=1-->no shuffle at all
dataset = dataset.shuffle(buffer_size=10)
# batch the dataset with each batch's size =2
# prefetch(1): typically add a small number of buffer size for prefetch can improve the performance by overlapping the preprocessing of data
dataset = dataset.batch(2).prefetch(1)
# print the shuffled batches of features & label in windows
for x, y in dataset:
    print("x=", x.numpy())
    print("y=", y.numpy())