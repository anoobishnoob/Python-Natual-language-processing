import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize


# print(nltk.__file__) very useful for just finding file paths


sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)
print(tok[5:15])

