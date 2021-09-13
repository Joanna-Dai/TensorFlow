import tensorflow_datasets as tfds

# load the entire train split dataset, which happen to be entire dataset for this case
data=tfds.load('cats_vs_dogs',split='train',as_supervised=True)

# notation syntax: [<start>:<stop>:<step>]
# load the first 1000 train records
data=tfds.load('cats_vs_dogs',split='train[:1000]', as_supervised=True)
# load the first 20% records
data=tfds.load('cats_vs_dogs',split='train[:20%]', as_supervised=True)
# load the last 1000 train records combined with first 1000 train records
data=tfds.load('cats_vs_dogs',split='train[-1000:]+train[:1000]', as_supervised=True)

# create own splits
# define train dataset as first 80% train records
train_data=tfds.load('cats_vs_dogs',split='train[:80%]', as_supervised=True)
# define test dataset as 10% of entire dataset
test_data=tfds.load('cats_vs_dogs',split='train[80%:90%]', as_supervised=True)
# define validation dataset as the last 10% of entire dataset
validation_data=tfds.load('cats_vs_dogs',split='train[-10%:]', as_supervised=True)

# count the number of records in the particular split to measure the length of dataset
train_length = [i for i, _ in enumerate(train_data)][-1] + 1
print(train_length)
