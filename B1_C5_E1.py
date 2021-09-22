import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
    ]

# take words and turn them into tokens
# tokenization: the process to use numbers to encode entire words
# num_words: the maximum number of tokens to generate from the corpus of words
# ovv_token: the new item to maintain sentence length and understand the context of the data containing previously unseen text
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
# fit_on_texts: create the tokenized word index, defaults to removing all punctuation except the apostrophe character
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
# encode the sentences into the sequences of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# use 0 to fill and get sentences into same shape and size
# pre-padding (default): add zeros at the beginning
# maxlen (default): length of longest sentence
# truncating (default): truncated from the beginning of the sentence
padded=pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
print(padded)



