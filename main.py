import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

df = pd.read_csv("Data.csv")
tokensDf = pd.DataFrame(index=df.index, columns=df.columns)



print(df.iloc[0, 0])