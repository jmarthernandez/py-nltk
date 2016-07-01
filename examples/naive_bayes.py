import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from pprint import pprint as pp

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# pp(all_words.most_common(15))
# pp(all_words['stupid'])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# pp((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
feature_sets = [(find_features(rev), category) for (rev, category) in documents]

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

# Naive Bayes
# posterior = prior occurences X (liklihood / evidence)

classifier = nltk.NaiveBayesClassifier.train(training_set)
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

# nltk algo
print('Naive Bayes:', nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(15)

# Sklearn algos
print('MNB_classifier:', nltk.classify.accuracy(MNB_classifier, testing_set) * 100)
print('BNB_classifier:', nltk.classify.accuracy(BNB_classifier, testing_set) * 100)
print('LogisticRegression_classifier:', nltk.classify.accuracy(LogisticRegression_classifier, testing_set) * 100)
print('SGDClassifier_classifier:', nltk.classify.accuracy(SGDClassifier_classifier, testing_set) * 100)
print('SVC_classifier:', nltk.classify.accuracy(SVC_classifier, testing_set) * 100)
print('LinearSVC_classifier:', nltk.classify.accuracy(LinearSVC_classifier, testing_set) * 100)
print('NuSVC_classifier:', nltk.classify.accuracy(NuSVC_classifier, testing_set) * 100)

