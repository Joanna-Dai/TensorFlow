# convert B1_C1 to TensorFlow Lite
# TensorFlow Lite:
#   part of tensorflow ecosystem and consists of tool needed for converting trained models and interpretting models
#   2 goals: 1) make models mobile-friendly (lightweight, low-latency) 2) provide a runtime for different mobile platforms (Android, iOS, mobile Linux like Raspberry Pi)
#   workflow overview: train model using tensorflow --> convert model to tensorflow lite format --> load and run it using tensorflow lite interpreter
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

#input data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
print("Here is what I learned: {}".format(l0.get_weights()))


# tf converter (SavedModel or Keras H5)

# step 1: specify directory and save the model to directory
export_dir='C:/Users/user/tensorflow_datasets/saved_model/B1_C1'
tf.saved_model.save(model, export_dir)

# step 2: covert and save the model
# convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
# save out tflite model using pathlib
import pathlib
tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)

# step 3: load the model and allocate the tensors
# load into interpreter and allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
# get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print to inspect
print(input_details)  # input shape=array[1,1]: 1 list that contains 1 value i.e. [[x]]
print(output_details) # output shape=array[1,1]: 1 list that contains 1 value i.e. [[y]]

# step 4: perform the prediction
to_predict = np.array([[10.0]], dtype=np.float32)
print(to_predict)
# get the interpreter to do the prediction
interpreter.set_tensor(input_details[0]['index'], to_predict) # input_details[0] means a single input and will address at the index
interpreter.invoke() # invoke the interpreter
# read the prediction
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)

# Results:
# [[10.]]
# [[18.979979]]
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.