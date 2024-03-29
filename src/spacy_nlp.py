
#!/Users/soraward/opt/miniconda3/bin/python3 
data_root = "../archive/"

# ML stuff
import re
import numpy as np
import torch
from sklearn.linear_model import Lasso
import pandas as pd
import statsmodels.api as sm
# from statsmodels.tsa.api import VAR

# for nlp
# FinBERT
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_classifier = pipeline("sentiment-analysis", model = finbert_model, tokenizer = finbert_tokenizer)
# spacy
import spacy
from spacy import displacy
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

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
# from nasdaq_analysis import nasdaq

cnbc_path = data_root + "cnbc_headlines.csv"
guardian_path = data_root + "guardian_headlines.csv"
reuters_path = data_root + "reuters_headlines.csv"
news_path = data_root + "news.csv"



# functions

def sentiment(text):
    result = finbert_classifier([text])[0]
    sent = 0
    if result["label"] == "nevative":
        sent = -result["score"]
    elif result["label"] == "positive":
        sent = result["score"]
    else:
        sent = 0.0
    return sent

def spacy_sentiment(text):
    tokens = nlp(text)
    return tokens._.polarity

# for targetting specific stock
def add_word(df, word, column_name="Headlines"):
    df_c = df.copy()
    df_c[word] = df_c[column_name].apply(lambda txt: any(txt_word == word for txt_word in str(txt).split()))
    return df_c

def df_word(df, word, column_name="Headlines"):
    df_c = add_word(df, word, column_name)
    return df_c.loc[df_c[word]]

def time_to_str(time):
    return time.strftime("%Y-%m-%d")

def str_to_time(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d")

# generate dates
def generate_dates(start, end):
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    date_generated = [(start + datetime.timedelta(days=x)).strftime("%Y-%m-%d") for x in range(0, (end-start).days)]
    return date_generated

def generate_dates_df(start, end, column_name="Time"):
    dates_df = pd.DataFrame({column_name:generate_dates(start,end)})
    dates_df[column_name] = pd.to_datetime(dates_df[column_name])
    dates_df.set_index(column_name, inplace=True)
    return dates_df

def news_by_source(df, source_name, column_name = "title"):
    if source_name:
        source_ = news.groupby("source").get_group(source_name)
    else:
        source_ = df
    source = source_[["title"]]
    return source

def prepare_nasdaq(nasdaq, start, end, stock_name="AAPL"):
    nasdaq = nasdaq.loc[nasdaq["Name"]==stock_name]
    dates = generate_dates_df(start, end)
    new_nasdaq = dates.merge(nasdaq, how="left", left_index=True, right_index=True)
    new_nasdaq.dropna(inplace=True)
    return new_nasdaq
    

################ merge with quantiative data ###################
def guar_news_nasdaq(guardian, news, nasdaq, columns_to_use=["Adj Close", "sentiment"]):
   
    # getting timeframe aligned, concat nlp data 
    start_guar = time_to_str(guardian.index[-1]) 
    end_guar = time_to_str(guardian.index[0]) 
    start_news = time_to_str(news.index[0]) 
    end_news = time_to_str(news.index[-1]) 

    start = max(start_news, start_guar)
    end = min(end_news, end_guar)
    guardian = guardian[start:end]
    news = news[start:end]
    # nasdaq = nasdaq[start:end]

    # concat two datas
    data = pd.concat([news, guardian], join="outer").sort_index()

    # merge for Apple data
    data_Apple = df_word(data, "Apple")[["Headlines"]]
    data_Apple["sentiment"] = data_Apple["Headlines"].apply(spacy_sentiment)
    data_Apple = data_Apple[["sentiment"]].groupby(data_Apple.index).mean()

    nasdaq_Apple = prepare_nasdaq(nasdaq, start, end, "AAPL")

    # since we are merging on nasdaq, we are not using the news from saturdays or sundays
    data_Apple = nasdaq_Apple.merge(data_Apple, how="left", left_index=True, right_index=True)

    data_Apple = data_Apple[columns_to_use]

    return data_Apple

################### without guardian data #####################
def news_nadaq(news, nasdaq, columns_to_use=["Adj Close", "sentiment"]):
    start_, end_ = time_to_str(news.index[0]), time_to_str(news.index[-1])
    news_Apple = df_word(news, "Apple")[["Headlines"]]
    news_Apple["sentiment"] = news_Apple["Headlines"].apply(spacy_sentiment)
    news_Apple = news_Apple[["sentiment"]].groupby(news_Apple.index).mean()
    # 小さな値と大きな値の差が激しいと感じたため、　三乗根をとった
    news_Apple = np.cbrt(news_Apple)

    news_nasdaq_Apple = prepare_nasdaq(nasdaq, start_, end_, "AAPL")

    news_Apple = news_nasdaq_Apple.merge(news_Apple, how="left", left_index=True, right_index=True)
    news_Apple = news_Apple[columns_to_use]

    return news_Apple


if __name__ == "__main__":
    ###################### nlp data, load everything ######################
    ## guardian is only headlines
    # guardian = pd.read_csv(guardian_path)
    # guardian.Time = pd.to_datetime(guardian.Time, errors="coerce")
    # guardian.dropna(inplace=True)
    # guardian.set_index("Time", inplace=True)
    # guardian.index.name = None

    # since we think any reuters data is included in news data, we only load news data
    news = pd.read_csv(news_path)#, nrows=100000)
    news["timestamp"] = news["timestamp"].apply(lambda string: string[:10])
    news["timestamp"] = pd.to_datetime(news["timestamp"])
    news.set_index("timestamp", inplace=True)
    news.index.name = None
    # renaming title to headlines
    news.rename(columns={"title":"Headlines"}, inplace=True)
    news = news[["Headlines"]]

    ##################### quantiative data #####################
    nasdaq = pd.read_csv(data_root + "NASDAQ_100_Data_From_2010.csv", sep="\t")
    # resetting index to datetime
    nasdaq.Date = pd.to_datetime(nasdaq.Date)
    nasdaq.set_index("Date", inplace = True)
    nasdaq["log_Volume"] = np.log(np.array(nasdaq["Volume"]+1e-9))

    # without guardian
    news_Apple = news_nadaq(news, nasdaq)

    news_Apple.sentiment.fillna(0.0, inplace=True)
    news_Apple = news_Apple.reindex(pd.date_range(start=news_Apple.index[0],\
                                    end=news_Apple.index[-1]), method="nearest")

    # talking log because Adj Close is too big
    news_Apple["Adj Close"] = np.log(news_Apple["Adj Close"]+1e-9).diff().diff()
    news_Apple.dropna(inplace=True)# dropping first row cause we took diff()

    # days to predict
    steps = 10
    forecast_idx = int(len(news_Apple.index)*0.7)
    forecastdate = news_Apple.index[forecast_idx]  # 2020-09-28

    train_df = news_Apple[news_Apple.index[0] : forecastdate - pd.offsets.Day(1)]
    pred_exog = news_Apple[forecastdate : forecastdate + pd.offsets.Day(steps - 1)]['sentiment']
    whole_endog = news_Apple.loc[news_Apple.index[0] : forecastdate + pd.offsets.Day(steps - 1)]["Adj Close"]

    fit_with_exog = sm.tsa.statespace.SARIMAX(endog = train_df["Adj Close"],
                                            exog = train_df["sentiment"],
                                            enforce_invertibility=False,
                                            order=(1, 1, 1)).fit()

    fit = sm.tsa.statespace.SARIMAX(train_df["Adj Close"],
                                enforce_invertibility=False,
                                order=(1, 1, 1)).fit()

    result_df = pd.DataFrame({
                            "Adj Close":whole_endog.iloc[forecast_idx-20:],
                            "pred w/ sentiment":fit_with_exog.forecast(steps=steps,exog = pred_exog),
                            "pred":fit.forecast(steps=steps),
                            "sentiment":pred_exog
                            }
                            )
    # print(forecastdate)
    result_df.plot(figsize=(15, 5))
    plt.title("prediction from:{}".format(forecastdate))
    plt.show()

    # やってみた考え：sentimentは一日遅れで影響しているように見えるところもある（微妙）
    # 次やること：逐次的に予測した結果を見る







    # from the calcuations below, we conclude (1, 1, 1) is the best

    # max_p = 3
    # max_d = 3
    # max_q = 3
    # num = 0
    # modelSelection = pd.DataFrame(index=range(max_p*(max_d+1)*(max_q+1)), columns=["model", "aic"])
    # for p in tqdm.tqdm(range(1, max_p+1)):
    #     for d in range(0, max_d+1):
    #         for q in range(0, max_q+1):
    #             arimax = sm.tsa.SARIMAX(endog, exog=exog, order=(p, d, q)).fit()
    #             modelSelection.iloc[num]["model"] = "order=(" + ", ".join(map(str, [p, d, q])) + ")"
    #             modelSelection.iloc[num]["aic"] = arimax.aic
    #             num = num + 1
    # print(modelSelection)
    # print(modelSelection.loc[modelSelection.aic == min(modelSelection.aic)])



