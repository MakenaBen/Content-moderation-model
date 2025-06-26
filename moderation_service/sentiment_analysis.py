from transformers import pipeline

sentiment_ model = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_model(text)
    return result[0]['label']