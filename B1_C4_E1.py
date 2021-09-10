import tensorflow as tf
import tensorflow_datasets as tfds

# inspect the dataset by printing the item in tfds.load
mnist_data = tfds.load("fashion_mnist")
# there are two splits in dataset: train and test
for item in mnist_data:
    print(item)

#load the train dataset by sepcifying split as "train"
mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

#insepct a value in the dataset, dictionary with two keys
for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    print(item['image'])
    print(item['label'])

#details about the contents of the dataset: description, homepage, data_path, features, keys, splits, citation
mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")
print(info)