#!/Users/soraward/opt/miniconda3/bin/python3 
data_root = "../archive/"

# ML stuff
import re
import numpy as np
# import torch
from sklearn.linear_model import Lasso
import pandas as pd

# for nlp
# FinBERT
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_classifier = pipeline("sentiment-analysis", model = finbert_model, tokenizer = finbert_tokenizer)

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["image.cmap"] = "cividis" # this doesn't seem to be working

# basic stuff
import datetime
import requests
import io
from collections import Counter

#local import nasdaq
from nasdaq_analysis import nasdaq

cnbc_path = data_root + "cnbc_headlines.csv"
guardian_path = data_root + "guardian_headlines.csv"
reuters_path = data_root + "reuters_headlines.csv"


def sentiment(text):
   result = finbert_classifier([text])
   sent = 0
   if result["label"] == "nevative":
      sent = -result["score"]
   elif result["label"] == "positive":
      sent = result["score"]
   else:
      sent = 0.0
   return sent
        
def df_word(df, word):
   df_c = df.copy()
   df[word] = df_c["Headlines"].apply(lambda tokens: any(token.text == word for token in tokens))
   del df_c
   return df

def time_to_str(time):
   return time.strftime("%Y-%m-%d")


#とりまロイターとguardianだけ(100行)
reuters_sm = pd.read_csv(reuters_path, nrows=100)
guardian_sm = pd.read_csv(guardian_path, nrows=200)

# change datetime format and make it to index
reuters_sm.Time = pd.to_datetime(reuters_sm.Time)
reuters_sm.set_index("Time", inplace=True)

guardian_sm.Time = pd.to_datetime(guardian_sm.Time, errors="coerce")
guardian_sm.dropna(inplace=True)
guardian_sm.set_index("Time", inplace=True)

# only using headlines, guardian is only headlines already
reuters_sm = reuters_sm["Headlines"]
start_reu = time_to_str(reuters_sm.index[0]) 
end_reu = time_to_str(reuters_sm.index[-1])
start_guar = time_to_str(guardian_sm.index[0]) 
end_guar = time_to_str(guardian_sm.index[-1])
start = max(start_reu, start_guar)
end = min(end_reu, end_guar)

nasdaq_sm = nasdaq[start:end]
reuters_sm = reuters_sm[start:end]
guardian_sm = guardian_sm[start:end]

data = pd.merge(reuters_sm, guardian_sm, how="outer", left_index=True, right_index=True)


# reuters_sm["sentiment"] = reuters_sm["Headlines"].apply(sentiment)
# guardian_sm["sentiment"] = guardian_sm["Headlines"].apply(sentiment)
# for sentiment on a stock on the same day, add them

print(data)