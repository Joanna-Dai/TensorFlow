import tensorflow as tf
import tensorflow_datasets as tfds

data, info=tfds.load("mnist", with_info=True)
print(info)

# after loading data per TFDS, data is downloaded and cached to TFRecord format
filename='C:/Users/user/tensorflow_datasets/mnist/3.0.1/mnist-test.tfrecord-00000-of-00001'

raw_dataset=tf.data.TFRecordDataset(filename)
# each record consisting of an integer indicating the length of the record, a cyclic redundancy check (CRC), a byte array of the data and a CRC of that byte array
# the record are concatenated into the file and then sharded in the case of large datasets
for raw_record in raw_dataset.take(1):
    print(raw_record)

# create a description of the features based on the feature info per print(info)
feature_description = {
    'image': tf.io.FixedLenFeature([], dtype=tf.string),
    'label': tf.io.FixedLenFeature([], dtype=tf.int64),
}

# parse the input `tf.Example` proto using the dictionary above
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)

# the result shows the image is a Tensor, stored as string, contains a PNG; the label is stored as int with value=2
# by reading the TFRecord, you can decode PNG via PNG decoder library like pillow
for parsed_record in parsed_dataset.take(1):
    print((parsed_record))
