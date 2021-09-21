import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
    ]

# take words and turn them into tokens
# tokenization: the process to use numbers to encode entire words
# num_words: the maximum number of tokens to generate from the corpus of words
tokenizer = Tokenizer(num_words = 100)
# fit_on_texts: create the tokenized word index, defaults to removing all punctuation except the apostrophe character
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
# encode the sentences into the sequences of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)


