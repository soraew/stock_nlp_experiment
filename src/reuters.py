#!/Users/soraward/opt/miniconda3/bin/python3 
data_root = "../archive/"


# ML stuff
import numpy as np
import torch
from sklearn.linear_model import Lasso
import pandas as pd

# for nlp
from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import spacy
from spacy import displacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob') # for semantic analysis

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["image.cmap"] = "cividis" # this doesn't seem to be working

# basic stuff
import datetime
import requests
import io
from collections import Counter

# load dataset
reuters_path = data_root + "reuters_headlines.csv"
reuters = pd.read_csv(reuters_path)

# change datetime format and make it to index
reuters.Time = pd.to_datetime(reuters.Time)
reuters.set_index("Time", inplace=True)

# reuters data is 2018-03-20 -> 2020-07-18


def get_polarity(df, n):
    df_iter = df.iterrows()
    for i in range(n):
        df_iter_ = next(df_iter)
        txt = df_iter_[1]["Headlines"]
        tokens = nlp(txt)
        print("txt          >>", txt)
        print("polarity     >>",tokens._.polarity)
        print("subjectivity >>",tokens._.subjectivity)
        print("assessments  >>", tokens._.assessments, "\n")
        
get_polarity(reuters, 100)

# tokenize reuters
reuters_sm = reuters[:1000].copy()
reuters_sm["Headlines"] = reuters_sm["Headlines"].apply(nlp)





