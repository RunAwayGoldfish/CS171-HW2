import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np

df = pd.read_csv("Data.csv")

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)


tokensDf = pd.DataFrame(index=df.index, columns=df.columns)
arr = df.to_numpy()
labels = df.columns.to_list()

emotions = ["Fear", "Anger", "Surprise", "Disgust", "Sadness", "Joy"]

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

def getPriors(shouldPrint = 0):
    priors = {}

    for i in emotions:
        priors[i] = 0



    a, b = arr.shape
    a = 29 # [0,29] Training

    emotionSentenceCounts = np.zeros(len(emotions))

    for i in range(len(emotionSentenceCounts)):
        for col in range(1,b,2):
            if(emotions[i].lower() in labels[col].lower()):
                for k in range(a):
                    text = arr[k, col]
                    if(isinstance(text, float)):
                        continue
                    sentences = sent_tokenize(text)
                    emotionSentenceCounts[i] += len(sentences)
                    
    totalSentences = 0
    for j in range(1,b,2):
        for i in range(a):
            text = arr[i, j]
            if(isinstance(text, float)):
                continue
            sentences = sent_tokenize(text)
            totalSentences += len(sentences)

    for i in range(len(emotionSentenceCounts)):
        priors[emotions[i]] = float(emotionSentenceCounts[i] / totalSentences)

    return priors

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
        
def getEmotionLiklihoods(emotion):
    column = -1
    for i in range(len(labels)):
        if((emotion + " Sentence").lower() in labels[i].lower()):
            column = i 
            break

    if(column == -1):
        print("No emotion: ", emotion)
        return

    vocabulary = getTotalVocabSet()
    emotionLikelihoods = {}
    for i in vocabulary:   
        emotionLikelihoods[i] = 1

   
    a, b = arr.shape
    a = 29 # Training

    for col in range(1,b,2):
        if(emotion.lower() in labels[col].lower()):
            #print(col)
            for k in range(a):
                text = arr[k, column]
                if(isinstance(text, float)):
                    continue
                words = word_tokenize(text)
                for word in words:
                    if(word in emotionLikelihoods):
                        emotionLikelihoods[word] += 1
                    else:
                        emotionLikelihoods[word] = 1  

    counter = 0
    for i in emotionLikelihoods.values():
        counter += i

    for key in emotionLikelihoods.keys():
        emotionLikelihoods[key] = emotionLikelihoods[key] / counter

    return emotionLikelihoods

def printDictNice(dictionary):
    for key in dictionary:
        print(f"{key} {dictionary[key]}")

def getBayesValue(sentence, emotion):
    column = -1
    for i in range(len(labels)):
        if((emotion + " Sentence").lower() in labels[i].lower()):
            column = i 
            break

    if(column == -1):
        print("No category: ", emotion)
        return

    liklihoods = getEmotionLiklihoods(emotion)

    sentence = sentence.lower()
    vocabulary = getTotalVocabSet()
    score = getPriors()[emotion]

    for i in sentence:
        if i in vocabulary:
            score += np.log(liklihoods[i])

    return score



def getBayesPrediction(sentence):
    sentence = sentence.lower()
    vocabulary = getTotalVocabSet()
    bestBayesValue = float('-inf')
    bayesPrediction = ""

    for i in range(len(emotions)):
        emotion = emotions[i].lower()

        bayesValue = getBayesValue(sentence, emotion)
        print(bayesValue, emotion)

        if(bayesValue > bestBayesValue):
            bayesPrediction = emotion
            bestBayesValue = bayesValue

    return bayesPrediction


#question3_1(1)


sentence = "As she hugged her daughter goodbye on the first day of college, she felt both sad to see her go and joyful knowing that she was embarking on a new and exciting chapter in her life."
#printDictNice(getEmotionLiklihoods("sadness"))
#print(getBayesValue(sentence, "sadness"))
print(getBayesPrediction(sentence))

#printDictNice(getPriors())


#print(getEmotionLiklihoods("Fear"))
#printDictNice(getEmotionLiklihoods("Fear"))

#question3_1()