import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

(training_images, training_labels), (test_images, test_labels) = data.load_data()
print(type(training_images))

#reshape: prior to normalizing image, add extra dimension (color channel) to the images
training_images=training_images.reshape(60000, 28, 28, 1)
training_images  = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

#tf.keras.layers.Conv2D: to implement a convolutional layer (filter that reduce the image to features)
#Conv2D(64,): learn 64 convolutions (i.e. randomly initialize and over time learn filter values that work best to match the input to their output)
#Conv2D(,(3,3),): 3x3 sized filter of weights to multiply a pixel with its neighbours to get a new value of that pixel (transform/reduce image to features)
#Conv2D(input_shape=(28,28,1)) 28x28 pixel with 1 color channel (3 RGB is for color image)
#tf.keras.layers.MaxPooling2D(2, 2): immediate after convolutional layer; split the image into 2x2 pools and picked the max value of each (to reduce amount of info)
#Flatten(): after covolutions and pooling, prior to dense layers, the data will be flattened
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model summary:
#1st conv layer: 28x28 --conv2D (3x3 filter, left/right/top/bottom pixel has no full neighours to conv)--> (28-2)x(28-2)
#2nd pooling layer: 26x26 --pooling (2x2 max) --> 26/2 x 26/2
#3rd conv layer: 13x13--conv23 (3x3 filter) --> (13-2) x (13-2)
#4th pooling layer: 11x11--pooling (2x2 max)--> rounddown(11/2) x roundown (11/2)

# parameters:
#1st conv layer: # para = 64 convolutions x (3x3 weights for each conv to learn+1 bias)=64x10=640
#2nd pooling layer: 13x13, 64 of them; # para = 0 (nothing to learn but just to reduce the image)
#3rd conv layer: # para = 64 convolutions x previous 64 conv x 9 para + 64 convolutions x 1 bias = 36928
#4th pooling layer: 5x5, 64 of them; # para=0 (nothing to learn but just to reduce the image)
#5th flatten: take images to numeric value
#6th dense layer: 128 neurons; # para = 128 x (5x5x64) +128 x 1 bias =204928
#7th dense layer: 10 neuron; # para  = 10 x 128 previous output + 10 x 1bias= 1290
#total # learned para = 640 + 0 + 36928 + 0 + 204928 + 1290 = 243786 --> the network requires us to learn the best set of 243786 parameters to match the input images to their labels

model.summary()

#observation: adding convolutions to neural network increases its ability & accuracy to classify images
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])