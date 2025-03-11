"""All the additional functions are written in this Python file, which are used through out this project"""

## This function is used for getting news data from Guardian API
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

def get_news_data(API_KEY: str, from_date: str, to_date:str):
    try:
      start = datetime.strptime(from_date, "%Y-%m-%d").date()
      end = datetime.strptime(to_date, "%Y-%m-%d").date()
      dates_generated = [start + timedelta(days=x) for x in range(0, (end-start).days+1)]
      # print("dates done")

      articles = []
      for days in tqdm(dates_generated):
          days = datetime.strftime(days, "%Y-%m-%d")

          BASE_URL = "https://content.guardianapis.com/search"
          params = {
              "api-key": API_KEY,
              "from-date": days,
              "to-date": days,
              "section": "business",
              # "sectionName": "business news",
              # "q": f"{dow_jones_companies[ticker]}",
              "show-fields": "headline, body",
              "order-by": "newest",
              "page-size": 100
          }

          response = requests.get(BASE_URL, params)
          if response is not None:
              data = response.json()

              for article in data["response"]["results"]:
                  articles.append({
                      "Title": article["webTitle"],
                      "URL": article["webUrl"],
                      "Publication Date": article["webPublicationDate"],
                  })
          else:
              pass

      df = pd.DataFrame(articles)
      df = df.set_index("Publication Date")

      return df

    except:
      df = pd.DataFrame(articles)
      df = df.set_index("Publication Date")
      print("Limit completed...")
      return df

news = get_news_data("API-KEY", "2023-04-07", "2025-02-27") # (api-key, from_date, to_date)



## This function is used to get sentiment scores of news data using FinBERT model
# Suggestion: Try to run with GPU and with higher RAM (use Colab)
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def calculate_compound_scores(probabilities):
    return probabilities['positive'] - probabilities['negative']

def get_news_sentiment(df):

    titles = list(df["Title"].values)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    inputs = tokenizer(titles, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits
    print("[Logits generated...]")
    print()

    probabilities = scipy.special.softmax(logits.numpy(), axis=1)
    labels = list(model.config.id2label.values())
    print("[Labels are listed...]")
    print()

    sentiment_scores = []
    for i, title in enumerate(titles):
        scores_dict = {labels[j]: probabilities[i][j] for j in range(len(labels))}
        sentiment_scores.append(scores_dict)
    print("[Sentiment scores are given...]")
    print()

    compound_scores = [calculate_compound_scores(scores) for scores in sentiment_scores]
    compound_scores = [float(x) for x in compound_scores]

    return compound_scores



## This function is used to combine different dataframes (stocks data & sentiment data)
import os
import pandas as pd

def merge_dataframes(directory: str):

    directory = os.fsencode(directory)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            # import Stock data and convert date format
            df1 = pd.read_csv("./stocks_data/" + filename)
            df1["Date"] = pd.to_datetime(df1["Date"], utc=True).dt.tz_convert(None).dt.date

            # import sentiment data and convert dateformat
            df2 = pd.read_csv("sentiment_avg.csv") # change "sentiment_avg.csv" with your sentimnet scores csv file
            df2["Publish Date"] = pd.to_datetime(df2["Publish Date"]).dt.date

            # merge dataframes
            df = pd.merge(df1, df2, left_on="Date", right_on="Publish Date")
            df = df.drop("Publish Date", axis=1)
            df.rename(columns={"Average": "Sentiment Average"}, inplace=True)

            # rewrite existing csv file with additional changes
            df.to_csv("./stocks_data/" + filename, index=False)

directory = "./stocks_data" # change it with your directory name
merge_dataframes(directory)



## Title