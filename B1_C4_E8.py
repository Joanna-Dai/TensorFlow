import tensorflow as tf
import tensorflow_datasets as tfds
import multiprocessing

train_data = tfds.load('cats_vs_dogs', split='train', with_info=True)
# create a list of files
file_pattern = f'C:/Users/user/tensorflow_datasets/cats_vs_dogs/4.0.0/cats_vs_dogs-train.tfrecord*'
# load the list of files into files via tf.data.Dataset.list_files
files = tf.data.Dataset.list_files(file_pattern)

# load the files into dataset
# parallelizing Extract process (with the maximum degree of parallelism)
train_dataset = files.interleave(tf.data.TFRecordDataset,
                                 # the number of input elements that are processed concurrently
                                 # mapping function will decode 4 records a time for the data loaded from disk
                                 # if not specified, it will be derived from number of CPU cores
                                 cycle_length=4,
                                 # the number of parallel calles to execute
                                 # tf.data.experimental.AUTOTUNE means value is dynamically set based on CPU
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE
                                 )

# parallelizing Transformation of data

# mapping function that loads the new TFRecord and coverts it to usable content
def read_tfrecord(serialized_example):
    feature_description={
    "image": tf.io.FixedLenFeature((), tf.string, ""),
    "label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    example = tf.io.parse_single_example(
    serialized_example, feature_description
    )
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (300, 300))
    return image, example['label']

# if not tf.data.experimental.AUTOTUNE, the num_parallel_calls can be count of CPUs
cores = multiprocessing.cpu_count()
print(cores)
train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
train_dataset = train_dataset.cache()

# prefetch based on the CPU cores that are available
train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# define the model
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

model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

# train the model
model.fit(train_dataset, epochs=10, verbose=1)

# summary: the extra code to parallelize the ETL process reduced the training time: 75 sec/epoch-->40 sec/epoch
