import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


example_text = "Hello there, how are you doing today? The weather is great and Python is awesome."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))


for i in word_tokenize(example_text):
    print(i)




