import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

# data.load_data(): give you an array of 60,000 28x28 training images and 10,000 28x28 test images
# training/test_labels: contain output Y as values from 0-9

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# each pixel (value 0-255) is represented by 0-1 (normalizing)
# normalization will improve tf performance
training_images  = training_images / 255.0
test_images = test_images / 255.0

#input X (image): grid size=28x28=784 pixel values(X), each pixel value ranges from 0-255
#output Y: 10 types of images (i.e. value 0-9)
#flatten: takes "square" value (2D array/images) and tunrs it into a line (1D array/numeric values)
#dense=256/128/64: 256/128/64 neurons to have have heir internal para randomly initialized
#dense=256/128/64: arbitrary number, more->slower, more->overfitting (+ve train data, -ve recog new), fewer->no sufficient para to learn
#dense=256/128/64: hyperparameter tunning
#activation=tf.nn.relu: execute on each neuron; relu stands for rectified linear unit, return a value if greater than 0 to less impact summing function
#dense=10: the output layer, end up with a prob. that te input pixels match that class
#activation=tf.nn.softmax: which one has the highest value

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    # introduce dropout each dense layer: randomly remove/ignore 20% neurons to reduce the chance of overspecialization for better generalization
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#adam: evolution of sgd, faster & more efficient given data amount
#loss: due to output is a category
#accuracy: return how often it correctly matched the input pixels to the output label

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#epochs: more can lead to higher accuracy but possible overfitting
#callbacks: stop training if accuracy >95% to avoid overfitting
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()
#incorporate callbacks with higher epochs
#model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks], validation_data=(test_images,test_labels))
#evaluate the model: pass trained model to have test data predict and sum up the result
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# model result pre-dropout: 95% accuracy for training set and 90% accuracy fo validation set, indicating slight overfitting
#                           (overspecialized neurons, neighbouring neurons end up with similar weight and bias)
# model result post-dropout:92% accuracy for training set and 90% accuracy for validation set
