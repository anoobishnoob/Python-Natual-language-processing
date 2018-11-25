import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import  SVC, LinearSVC, NuSVC
# # naive Bayes algorithm
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]
# the 3000 just means use up to the first 3000 words to check against
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
#  print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#  pos = prior ocurences x liklihood / evidence

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()





print("Original Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100,"%")
#  this is getting multilied by 100 so the number will be in a percent form
# classifier.show_most_informative_features(100)

# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
# this part is just creating it so I am commenting it out so it doesn't rewrite the data again, but keeping as a comment so I wont forget

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB Algo accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100,"%")

# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("GNB Algo accuracy:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100,"%")


# LogisticRegression, SGDClassifier
# from sklearn.svm import  SVC, LinearSVC, NuSVC


# probably should put all these lines in a for loop, but for now just playing around with this
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algo accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100,"%")


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Algo accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100,"%")

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Algo accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100,"%")

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Algo accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100,"%")


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Algo accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100,"%")