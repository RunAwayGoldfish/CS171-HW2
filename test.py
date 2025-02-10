import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

#nltk.download('punkt')

text = "This is a test sentence!"

sentences = sent_tokenize(text)
print("Sentences:", sentences)

words = word_tokenize(text)
print("Words:", words)