
import tensorflow_datasets as tfds

# load IMDb subword datasets (8k or 32k): no need to break up the sentences by word, has the encoders and decoders used to split and encode the corpus
# subword: medium between splitting corpus into letters (few tokens, low semantic) and words (more tokens, high semantic), useful to train a classifier fo language
(train_data,test_data),info=tfds.load('imdb_reviews/subwords8k',split=(tfds.Split.TRAIN, tfds.Split.TEST),as_supervised=True, with_info=True)

#access info object in text format
encoder=info.features['text'].encoder
#voc is made up of 8185 tokens
print('Vocabulary size: {}'.format(encoder.vocab_size))
#list of subwords (stopwords, punctutaiton, and grammar are all in the corpus, spaces are represented by "_")
print(encoder.subwords)

#encode string via above encoder
sample_string='Today is a sunny day'
#result: Encoded string is [6427, 4869, 9, 4, 2365, 1361, 606]
encoded_string=encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))
#check the first two tokens
print(encoder.subwords[6427-1]+" "+encoder.subwords[4869-1])
#decode the encoded string
original_string=encoder.decode(encoded_string)
test_string=encoder.decode([6427, 4869, 9, 4, 2365, 1361, 606])
print(original_string)
print(test_string)