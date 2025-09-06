# Review Sentiment Analysis Tool

This Python project analyzes customer review text and classifies sentiment using VADER, TextBlob, and OpenAI GPT models. It provides both a sentiment label (positive, negative, neutral) and a normalized score out of 5 for VADER and TextBlob. The OpenAI integration uses the latest GPT model for advanced sentiment classification.

## Features
- Multiple sentiment engines: VADER, TextBlob, OpenAI GPT (gpt-5-nano)
- Score normalization to a 0–5 scale for VADER and TextBlob
- Flexible user input for method selection
- Secure OpenAI API key management
- System prompt for context-aware GPT analysis

## Requirements
- Python 3.13+
- `nltk`, `textblob`, `openai`
- Download NLTK VADER lexicon (handled automatically)
- OpenAI API key (for GPT analysis)

## Usage
1. Install dependencies:
   ```powershell
   pip install nltk textblob openai
   ```
2. Run the script:

3. Enter your review and select the analysis method (`vader`, `textblob`, `openai`, or `all`).
4. For OpenAI, enter your API key when prompted (saved for future use).

## Output
- Sentiment label and score out of 5 for VADER and TextBlob
- Detailed sentiment classification from OpenAI GPT (gpt-5-nano)

## Example
```
Enter review: The product quality is excellent and delivery was fast.
Choose method (vader, textblob, openai, all): all

Sentiment Results
VADER: positive (score: 4.75/5)
TextBlob: positive (score: 4.5/5)
OpenAI: {
  "price": "neutral",
  "quality": "positive",
  "delivery": "positive",
  "customer_support": "not mentioned"
}
```


