import nltk
import random
from nltk.corpus import movie_reviews
from pprint import pprint as pp

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
pp(nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(15)
