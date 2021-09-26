from bs4 import BeautifulSoup
import string

sentence = 'I really want to try <ac> <jd>'

# 3 steps to clean up texts

# step 1: remove HTML tags
soup = BeautifulSoup(sentence)
sentence = soup.get_text()

# step 2: remove punctuations (to avoid fooling stopwords)
# step 3: remove the stopwords (too common and don't add any meaning) from the sentences
stopwords = ["a", "about", "above", "yours", "yourself", "yourselves"]
table = str.maketrans('', '', string.punctuation)
words = sentence.split()
filtered_sentence = ""
for word in words:
    word = word.translate(table)
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "

sentences.append(filtered_sentence)

print(sentence)