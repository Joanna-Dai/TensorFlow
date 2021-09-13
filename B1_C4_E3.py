import tensorflow as tf
import tensorflow_datasets as tfds

# for comparison: B1_C3_E2
# load the data of train split
train_data = tfds.load('horses_or_humans', split='train', as_supervised=True)
# load a seperate data set from tfds for validation
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)

# the training set is batched and shuffled to make the training more effective
train_batches = train_data.shuffle(100).batch(10)
# the validation set is batched
validation_batches=val_data.batch(32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# specify the number of validation steps to use per epoch (set it to 1 if not sure)
history = model.fit(train_batches, epochs=10, validation_data=validation_batches, validation_steps=1)