import os
import json
import openai
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# download vader if not present
nltk.download("vader_lexicon", quiet=True)

# helpers
def load_api_key():
    if os.path.exists("openai_key.txt"):
        with open("openai_key.txt", "r") as f:
            return f.read().strip()
    return None

def save_api_key(key):
    with open("openai_key.txt", "w") as f:
        f.write(key.strip())

def build_prompt(review):
    return f"""
Analyze the review and classify sentiment for:
- Price
- Quality
- Delivery
- Customer Support

Rules:
positive, negative, neutral, not mentioned

Return only valid JSON with keys:
{{
  "price": "...",
  "quality": "...",
  "delivery": "...",
  "customer_support": "..."
}}

Review: "{review}"
"""

def openai_sentiment(review, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = build_prompt(review)
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for sentiment analysis."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

def textblob_sentiment(review):
    score = TextBlob(review).sentiment.polarity
    # Convert polarity (-1 to 1) to score out of 5
    score_5 = round((score + 1) * 2.5, 2)
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": score_5}

def vader_sentiment(review):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(review)["compound"]
    # Convert compound (-1 to 1) to score out of 5
    score_5 = round((score + 1) * 2.5, 2)
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": score_5}

# main
if __name__ == "__main__":
    review = input("Enter review: ").strip()
    method = input("Choose method (vader, textblob, openai, all): ").strip().lower()

    api_key = None
    if method in ["openai", "all"]:
        api_key = load_api_key()
        if not api_key:
            api_key = input("Enter OpenAI API key: ").strip()
            save_api_key(api_key)

    print("\nSentiment Results\n")

    if method in ["vader", "all"]:
        vader_result = vader_sentiment(review)
        print(f"VADER: {vader_result['label']} (score: {vader_result['score']}/5)")

    if method in ["textblob", "all"]:
        tb_result = textblob_sentiment(review)
        print(f"TextBlob: {tb_result['label']} (score: {tb_result['score']}/5)")

    if method in ["openai", "all"]:
        result = openai_sentiment(review, api_key)
        print("OpenAI:", json.dumps(result, indent=2))
