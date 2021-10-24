import tensorflow as tf
import tensorflow_datasets as tfds

imdb_sentences = []
# by load into tfds.as_numpy, the text will be loaded as string not tensors
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
# create sentences and put them into imdb_sentences
for item in train_data:
 imdb_sentences.append(str(item['text']))

# create tokenizer and sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

# the tockenizer lists in order of frequency in dataset, so stopwords come first
print(tokenizer.word_index)