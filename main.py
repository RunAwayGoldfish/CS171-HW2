import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np

df = pd.read_csv("Data.csv")
tokensDf = pd.DataFrame(index=df.index, columns=df.columns)
arr = df.to_numpy()
labels = df.columns.to_list()


def countTokensDF():
    count = 0
    for index, row in df.iterrows():
        for col in df.columns:
            text = str(row[col])
            if(text == 'nan'):
                continue
            words = word_tokenize(text)
            count += len(words)
    print(count)

def countTokensNP():
    a, b = arr.shape
    count = 0
    for i in range(a):
        for j in range(b):
            text = arr[i, j]
            if(isinstance(text, float)):
                continue
            words = word_tokenize(text)
            count += len(words)
    print(count)

def tokenizedSentences():
    rows, cols = df.shape
    for i in range(rows):
        for j in range(cols):
            text = df.iloc[0, 1]

            sentences = sent_tokenize(text)
            words = word_tokenize(text)

            print(words)
            break
        break

def question3_1(shouldPrint = 0):
    a, b = arr.shape
    count = 0

    emotionSentenceCounts = np.zeros(int(b/2))
    priors = np.zeros(int(b/2))
    emotionId = 0

    totalSentences = 0
    for j in range(1,b,2):
        for i in range(a):
            text = arr[i, j]
            if(isinstance(text, float)):
                continue
            sentences = sent_tokenize(text)
            emotionSentenceCounts[emotionId] += len(sentences)
            totalSentences += len(sentences)

        emotionId += 1

    for i in range(len(emotionSentenceCounts)):
        priors[i] = emotionSentenceCounts[i] / totalSentences

        if(shouldPrint):
            print(f"{labels[i]} {priors[i]:.4f}")

 def question3_2():        
    pass



    


question3_1(1)