# using NASA weather data
# download time series weather csv data of SeaTac airport from https://data.giss.nasa.gov/cgi-bin/gistemp/stdata_show_v4.cgi?id=USW00024233&ds=14&dt=1

import numpy as np

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

# split into training and validation sets
split_time = 792
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# windowed dataset

