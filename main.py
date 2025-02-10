import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np

df = pd.read_csv("Data.csv")

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)


tokensDf = pd.DataFrame(index=df.index, columns=df.columns)
arr = df.to_numpy()
labels = df.columns.to_list()

training = 30-1





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
    a = 29 # [0,29] Training

    count = 0

    emotionSentenceCounts = np.zeros(int(b/2))
    priors = np.zeros(int(b/2))
    emotionId = 0

    totalSentences = 0
    for j in range(1,b,2):
        for i in range(a):
            text = arr[i, j]
            print(text)
            if(isinstance(text, float)):
                continue
            sentences = sent_tokenize(text)
            emotionSentenceCounts[emotionId] += len(sentences)
            totalSentences += len(sentences)
        break
        emotionId += 1

    for i in range(len(emotionSentenceCounts)):
        priors[i] = emotionSentenceCounts[i] / totalSentences

        if(shouldPrint):
            print(f"{labels[i]} {priors[i]:.4f}")

def getTotalVocabSet():
    vocabulary = set()

    a, b = arr.shape
    a = 29  # [0,29] Training
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
            words = word_tokenize(text)
            for word in words:
                vocabulary.add(word)

    return vocabulary
        
def getEmotionLiklihoods(category):
    column = -1
    for i in range(len(labels)):
        if((category + " Sentences").lower() == labels[i].lower()):
            column = i 
            break

    if(column == -1):
        print("No category: ", category)
        return

    vocabulary = getTotalVocabSet()
    categoryLikelihoods = {}
    for i in vocabulary:   
        categoryLikelihoods[i] = 1

   
    a, b = arr.shape
    a = 29 # Training
    
    for i in range(b):
        text = arr[i, column]
        if(isinstance(text, float)):
            continue
        words = word_tokenize(text)
        for word in words:
            if(word in categoryLikelihoods):
                categoryLikelihoods[word] += 1
            else:
                categoryLikelihoods[word] = 1

    counter = 0
    for i in categoryLikelihoods.values():
        counter += i

    for key in categoryLikelihoods.keys():
        categoryLikelihoods[key] = categoryLikelihoods[key] / counter

    return categoryLikelihoods

def printDictNice(dictionary):
    for key in dictionary:
        print(f"{key} {dictionary[key]}")

def getBayesValue(sentence, emotion):
    column = -1
    for i in range(len(labels)):
        if((emotion + " Sentences").lower() == labels[i].lower()):
            column = i 
            break

    if(column == -1):
        print("No category: ", category)
        return

    liklihoods = getEmotionLiklihoods(emotion)

    sentence = sentence.lower()
    vocabulary = getTotalVocabSet()
    score = 0

    for i in sentence:
        if i in vocabulary:
            score += np.log(liklihoods[i])

    return score



def getBayesPrediction(sentence):
    sentence = sentence.lower()
    vocabulary = getTotalVocabSet()

    for i in range(1, len(labels), 2):
        label = labels[i]
        index = label.find(" ")


        emotion = label[:index]
        print(emotion)

        



#question3_1()


sentence = "As she hugged her daughter goodbye on the first day of college, she felt both sad to see her go and joyful knowing that she was embarking on a new and exciting chapter in her life."
#printDictNice(getEmotionLiklihoods("sadness"))
#print(getBayesValue(sentence, "sadness"))
getBayesPrediction(sentence)

