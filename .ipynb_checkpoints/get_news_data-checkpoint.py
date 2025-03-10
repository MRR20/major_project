import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import scipy

API_KEY = "75e3c8c0-28e6-4166-961c-a72883c8ea3a"
BASE_URL = "https://content.guardianapis.com/search"

def news_data(API_KEY, BASE_URL):
    params = {
            "api-key": API_KEY,
            "from_date": "2025-02-23",
            "to-date": "2025-02-23",
            "q": "stocks OR trading OR finance OR business OR Apple OR Google OR Microsoft",
            "show-fields": "headline, body",
            "order-by": "newest",
            # "page-size": 50,
            }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    articles = []
    for article in data["response"]["results"]:
        articles.append({
            "Title": article["webTitle"],
            "URL": article["webUrl"],
            "Publication Data": article["webPublicationDate"],
            "Thumbnail": article["fields"].get("thumbnail", "No Image"),
            })

    df = pd.DataFrame(articles)

    return df["Title"]


def sentiment_analyzer():
    news_titles = news_data(API_KEY=API_KEY, BASE_URL=BASE_URL)

    news_titles = list(news_titles.values)
    print(news_titles)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    inputs = tokenizer(news_titles, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = scipy.special.softmax(logits.numpy(), axis=1)
    labels = list(model.config.id2label.values())

    sentiment_scores = []
    for i, title in enumerate(news_titles):
        scores_dict = {labels[j]: probabilities[i][j] for j in range(len(labels))}
        sentiment_scores.append(scores_dict)

    compound_scores = [calculate_compound_scores(scores) for scores in sentiment_scores]
    print(compound_scores)


def calculate_compound_scores(probabilities):
    return probabilities['positive'] - probabilities['negative']


sentiment_analyzer()