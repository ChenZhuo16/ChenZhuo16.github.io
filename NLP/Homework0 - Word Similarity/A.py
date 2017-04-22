import nltk
import sys

greeting = "Hello, world!"
print greeting

token_list = nltk.word_tokenize(greeting)
print "The tokens in the greeting are"
for token in token_list:
    print token
