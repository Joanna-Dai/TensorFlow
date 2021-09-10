import tensorflow as tf
import tensorflow_datasets as tfds

# for comparison: B1_C2
# tfds.as_numpy to covert tfds dataset for model.fit (while Keras datasets gave ndarray types that work natively in model.fit)
(training_images, training_labels), (test_images, test_labels) =\
    tfds.as_numpy(tfds.load('fashion_mnist', split=['train', 'test'], batch_size=-1, as_supervised=True))

# normalization
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)