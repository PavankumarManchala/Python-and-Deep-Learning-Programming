
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

from text_classification import twenty_train

tfidf_Vect = TfidfVectorizer()
tfidf_Vect1 = TfidfVectorizer(ngram_range=(1, 2))
tfidf_Vect2 = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
X_train_tfidf1 = tfidf_Vect1.fit_transform(twenty_train.data)
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train.data)

print("\n MultiNomialNB \n")

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

clf1 = MultinomialNB()
clf1.fit(X_train_tfidf1, twenty_train.target)

clf2 = MultinomialNB()
clf2.fit(X_train_tfidf2, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = round(metrics.accuracy_score(twenty_test.target, predicted), 4)
print("MultinomialNB accuracy is: ", score)


twenty_test1 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf1 = tfidf_Vect1.transform(twenty_test.data)

predicted1 = clf1.predict(X_test_tfidf1)

score1 = round(metrics.accuracy_score(twenty_test1.target, predicted1), 4)
print("MultinomialNB accuracy when using bigram is: ", score1)

twenty_test2 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf2 = tfidf_Vect2.transform(twenty_test.data)

predicted2 = clf2.predict(X_test_tfidf2)

score2 = round(metrics.accuracy_score(twenty_test2.target, predicted2), 4)
print("MultinomialNB accuracy when adding the stop-words is: ", score2)

print("============= SVM ============")
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(X_train_tfidf, twenty_train.target)

acc_svc = round(svc.score(X_train_tfidf, twenty_train.target) * 100, 2)
print("SVM accuracy is:", acc_svc / 100)