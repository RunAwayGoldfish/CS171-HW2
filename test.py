from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB



def test():
    emotions = ["Fear", "Anger", "Surprise", "Disgust", "Sadness", "Joy"]

    df = pd.read_csv("Data.csv")
    df = df.applymap(lambda x: re.sub(r'\s+', ' ', x.lower()) if isinstance(x, str) else x)
    tokensDf = pd.DataFrame(index=df.index, columns=df.columns)
    arr = df.to_numpy()
    labels = df.columns.to_list()

    a, b = arr.shape

    X = []
    Y = []

    for j in range(1,b,2):
        labelEmotions = set(word_tokenize(labels[j])[:-1])
        if("+" in labelEmotions):
            labelEmotions.remove("+")
        for i in range(a):
            text = arr[i, j]
            if(isinstance(text, float)):
                continue
            sentences = sent_tokenize(text)
            for sentence in sentences:
                for k in labelEmotions:
                    X.append(sentence)
                    Y.append(k)

            #return X, Y

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    y = Y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    nb = MultinomialNB()
    y_pred = nb.fit(X_train, y_train).predict(X_test)

    print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))


test()