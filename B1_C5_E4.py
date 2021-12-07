from bs4 import BeautifulSoup
import string
import tensorflow as tf
import tensorflow_datasets as tfds

#create a list of stopwords
#stopwords = []
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# translate by not fooling stopwords remover, string.punctuation contains a list of common punctuation marks
# (e.g. "it;" will be converted to "it"; "you'll" will be "youll")
table = str.maketrans('', '', string.punctuation)

imdb_sentences = []

#imdb_reviews, a dataset of 50,000 labeled movie reviews from the internet movie database with +ve / -ve in sentiment
#as_numpy will ensure the loaded data to be strings, not tensors
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    #the sentences are converted to lowercase given stopwords are stored in lowercase
    sentence = str(item['text'].decode('UTF-8').lower())

    #remove HTML tags such as <br>
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()

    #remove stopwords from the given list
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        #remove the punctuation
        word = word.translate(table)

        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "

    imdb_sentences.append(filtered_sentence)

#create a tokenizer object with maximum 5000 tokens/unique words from the corpus of the words
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50)

#create the tokenized word index
tokenizer.fit_on_texts(imdb_sentences)

#pass tokens to list of sentences and give back a list sequences of tokens
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index[0:20])
