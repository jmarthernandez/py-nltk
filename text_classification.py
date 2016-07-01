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

def run_print(label, classifier):
    print(label, nltk.classify.accuracy(classifier, testing_set) * 100)

NBclassifier = nltk.NaiveBayesClassifier.train(training_set)

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
run_print('Naive Bayes:', NBclassifier)
NBclassifier.show_most_informative_features(15)

# Sklearn algos
run_print('MNB_classifier:', MNB_classifier)
run_print('BNB_classifier:', BNB_classifier)
run_print('LogisticRegression_classifier:', LogisticRegression_classifier)
run_print('SGDClassifier_classifier:', SGDClassifier_classifier)
run_print('SVC_classifier:', SVC_classifier)
run_print('LinearSVC_classifier:', LinearSVC_classifier)
run_print('NuSVC_classifier:', NuSVC_classifier)

