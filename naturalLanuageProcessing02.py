from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "this is an example showing off stop word filtration"
stop_words = set(stopwords.words("english"))

# print(stop_words)

words = word_tokenize(example_sentence)

# filterered_sentence = []

# for w in words:
#    if w not in stop_words:
#        filterered_sentence.append(w)

# same one line expression
filterered_sentence = [w for w in  words if not w in stop_words]
print(filterered_sentence)