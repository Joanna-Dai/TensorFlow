import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

tokenizer = Tokenizer()
# create a string of all texts as input data
data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
# lower capital and use "\n" for line breaks
corpus = data.lower().split("\n")
# tokenize the word corpus
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

# split the sentence/line into multiple smaller sequences
input_sequences = []
# loop each line of the song
for line in corpus:
    # tokenize the line and change list "[[ ]]" to "[ ]" by putting [0]
	token_list = tokenizer.texts_to_sequences([line])[0]
    # loop smaller sequences of the line and start from (word 0 + word 1)
	for i in range(1, len(token_list)):
        # start from first two words and end with full line, note for list, list[:n+1]-->[0,n+1)-->[0,n]
		n_gram_sequence = token_list[:i+1]
        # append batch of smaller sequences of each line to the input_sequences
		input_sequences.append(n_gram_sequence)

# print full input_sequences
print(input_sequences)
# print smaller sequences of the first line
print(input_sequences[:7])

# pad all input_sequences into a regular shape
max_sequence_len = max(len(x) for x in input_sequences) # find the longest sentence in the input sequences first
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# print padded smaller sequences of the first line
print(input_sequences[:7])
# 453 smaller sequences in total
print(len(input_sequences))

# split smaller sequences into features (the sequence without last word) and label (the last word of the sequence)
# input_sequences[:, :-1]: list of input_sequences without last words of each input_sequence
# input_sequences[:, -1]: list of the last words of each input_sequences
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
# encode labels into a set of Ys (given output is a set of labels) that can be used to train (can eat memory very quickly)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# the example small sentence
print(input_sequences[5])
# the x of example small sentence
print(xs[5])
# the label of example small sentence
print(labels[5])
# the corresponding y of example small sentence
print(ys[5])
# same with the total number of smaller sequences
print(len(ys))

# model hierarchy
model = Sequential()
# embedding layer: vocab_size=total_words, embedding_dim=8
model.add(Embedding(total_words, 8))
# bidirectional LSTM as RNN layer: steps # = max_seq_len - 1 label at the end
model.add(Bidirectional(LSTM(max_sequence_len-1)))
# dense layer: neurons # = total words, activated by softmax (each neuron will be the prob. that the next word matches the word for that index value)
model.add(Dense(total_words, activation='softmax'))

# model guess approach
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# given the data is not a lot, model can be trained for a long time (1500 epoch)
# result: accuracy=93%, loss=0.26
# --> if we have a string of text already seen will predict the next word accurately about 95% of the time. If not previously seen, despite good accuracy, the network will still end up producing nonsensical text
# verbose = 0: will show you nothing (silent)
# verbose=1 will show you an animated progress bar like this: [============]
# verbose=2 will just mention the number of epoch like this: Epoch 1499/1500
history = model.fit(xs, ys, epochs=1500, verbose=1) #1500

# predicting the next word
# creating a phrase (initial expression on which the network will base all the content it generates)
seed_text = "in the town of athy"
# tokenize the seed text (texts_to_sequences returns an array even there's only one sentence/value, so use [0] to take the first element (i.e. the sentence))
token_list = tokenizer.texts_to_sequences([seed_text])[0]
# pad the seed sequence into the same shape
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
# model.predict: predict the next word for the padded token list and it will return the prob. for each word in the corpus
# np.argmax: pass the prob. to np.argmax to get the most likely one (axis=-1 means max. the last dimension of data shape)
predicted = np.argmax(model.predict(token_list), axis=-1)
# result: 68
print(predicted)
# search through the word index items until find predicted an print it out
# result: the most likely word (predicted by model) for the seed text is one
for word, index in tokenizer.word_index.items():
	if index == predicted:
		print(word)
		break

