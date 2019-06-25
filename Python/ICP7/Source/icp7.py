import requests
# import urllib.request
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from text_classification import twenty_train

wiki = "https://en.wikipedia.org/wiki/Google"
page = requests.get(wiki).text
soup = BeautifulSoup(page, "html.parser")

txt = str(soup.encode("UTF-8"))
file = open("input.txt",'w')
file.write(txt)
file.close()

nltk.download('punkt')

print("\n Sentence Tokenize")
s_tokens = nltk.sent_tokenize(txt)
print(s_tokens[:50])
print("\n Word tokenize")
w_tokens = nltk.word_tokenize(txt)
print(w_tokens[:50])

# nltk.download('averaged_perceptron_tagger')
# print("\n Parts of speech")
# pos = nltk.pos_tag(w_tokens)
# print(pos)
#
# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer
# from nltk.stem import SnowballStemmer
#
# pStemmer = PorterStemmer()
# lStemmer = LancasterStemmer()
# sStemmer = SnowballStemmer('english')
#
# print("Stemming")
# for w in w_tokens[:50]:
#     print(pStemmer.stem(w),
#           lStemmer.stem(w),
#           sStemmer.stem(w))
#
# print("Lemmatization")
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# print("\nLEMMATIZATION\n")
# for t in w_tokens[:50]:
#     print("Lemmatizer:", lemmatizer.lemmatize(t), ",    With POS=n:", lemmatizer.lemmatize(t, pos="n"))
#

# print("\n Trigram \n")
# from nltk.util import ngrams
# token = nltk.word_tokenize(txt)
# for s in s_tokens[:50]:
#      token = nltk.word_tokenize(s)
#      bigrams = list(ngrams(token, 2))
#      trigrams = list(ngrams(token, 3))
#      print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)

print("\n Named Entity Recognition \n")

from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')
for word in s_tokens[:50]:
    print(ne_chunk(pos_tag(word_tokenize(word))))

