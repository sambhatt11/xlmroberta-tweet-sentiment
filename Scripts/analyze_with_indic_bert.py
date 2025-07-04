import re
import torch
import pandas as pd
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util


# Clean text (supports Indian languages + English)
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emails
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '', text)
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F7FF"  # alchemical symbols
        u"\u2600-\u26FF"          # miscellaneous symbols
        u"\u2700-\u27BF"          # dingbats
        u"\u200c-\u200d"          # zero-width joiners
        u"\uFEFF"                  # BOM
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Keep valid characters from major Indian scripts
    cleaned = re.sub(
        r'[^\u0020-\u007E\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF\u0600-\u06FF]+',
        ' ', text
    )
    return ' '.join(cleaned.split()).strip()


# Detect language (optional)
def detect_language(text):
    try:
        return detect(text[:50])
    except:
        return 'unknown'


# Get sentiment label
def get_sentiment(model, tokenizer, text):
    if not text.strip():
        return "neutral", 0.0
    inputs = tokenizer(text[:512], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    labels = ['negative', 'neutral', 'positive']
    return labels[probs.index(max(probs))], max(probs)


# Get similarity between tweet and reply
def get_similarity(sim_model, tweet_text, reply_text):
    if not tweet_text or not reply_text:
        return 0.0
    embeddings = sim_model.encode([tweet_text, reply_text])
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(cosine_sim, 2)


# Zero-shot classification for custom labels
def classify_reply(zero_shot_pipe, reply_text):
    if not reply_text.strip():
        return "neutral"
    candidate_labels = ["hate","criticism","support","appreciation","insult","celebration"]
    result = zero_shot_pipe(reply_text, candidate_labels)
    return result['labels'][0]


# Load tweets and replies safely
def load_tweets_and_replies(tweets_file='5856457891030680519_tweet.csv', replies_file='5856457891030680519_replies.csv'):
    print("📥 Loading tweets and replies datasets...")

    try:
        df_tweets = pd.read_csv(tweets_file, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Error loading tweets.csv: {e}")
        return None, None

    try:
        df_replies = pd.read_csv(replies_file, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Error loading replies.csv: {e}")
        return None, None

    # Ensure required columns exist
    if 'tweet_id' not in df_tweets.columns:
        df_tweets.rename(columns={df_tweets.columns[0]: 'tweet_id'}, inplace=True)
    if 'tweet_text' not in df_tweets.columns:
        df_tweets.rename(columns={df_tweets.columns[1]: 'tweet_text'}, inplace=True)

    if 'reply_text' not in df_replies.columns:
        df_replies.rename(columns={df_replies.columns[0]: 'reply_text'}, inplace=True)

    # Convert tweet_id to int and fix scientific notation
    df_tweets['tweet_id'] = pd.to_numeric(df_tweets['tweet_id'], errors='coerce').astype('Int64')
    df_replies['tweet_id'] = pd.to_numeric(df_replies['tweet_id'], errors='coerce').astype('Int64')

    return df_tweets, df_replies


# Main function
def analyze_replies_with_context(tweet_csv_path="5856457891030680519_tweet.csv", replies_csv_path="5856457891030680519_replies.csv", output_file="test_replies_with_analysis.csv"):
    df_tweets, df_replies = load_tweets_and_replies(tweet_csv_path, replies_csv_path)

    if df_tweets is None or df_replies is None:
        print("❌ Failed to load datasets.")
        return

    print("🧠 Loading models...")

    # Sentiment model
    sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)
    sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    # Semantic similarity model
    sim_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Zero-shot classifier
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Pre-cache tweets
    tweet_cache = dict(zip(df_tweets['tweet_id'], df_tweets['tweet_text']))

    results = []

    for idx, row in enumerate(df_replies.itertuples(index=False), 1):
        raw_text = getattr(row, 'reply_text', '') or ''
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        tweet_id = getattr(row, 'tweet_id', None)
        if pd.isna(tweet_id) or tweet_id not in tweet_cache:
            print(f"⚠️ No matching tweet found for reply ID {tweet_id}")
            continue

        tweet_text = tweet_cache[tweet_id]

        try:
            # Get sentiment
            sentiment, confidence = get_sentiment(sent_model, sent_tokenizer, cleaned)

            # Get similarity
            similarity_score = get_similarity(sim_model, tweet_text, cleaned)

            # Get custom label
            custom_label = classify_reply(zero_shot, cleaned)

            results.append({
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "tweet_text": tweet_text,
                "reply_username": getattr(row, 'reply_username', ''),
                "reply_date": getattr(row, 'reply_date', ''),
                "tweet_id": tweet_id,
                "language": detect_language(cleaned),
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "relevance_score": similarity_score,
                "custom_label": custom_label
            })

            print(f"[{idx}] 🧠 {sentiment} | {custom_label} | {cleaned[:50]}...")

        except Exception as e:
            continue

    if not results:
        print("❌ No valid replies were processed.")
        return

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved {len(result_df)} replies with analysis to '{output_file}'")

    # Summary stats
    total = len(result_df)
    pos = len(result_df[result_df['sentiment'] == 'positive'])
    neg = len(result_df[result_df['sentiment'] == 'negative'])
    neu = total - pos - neg

    print("\n📊 Public Reply Sentiment Breakdown")
    print(f"👍 Positive: {pos} ({pos / total * 100:.1f}%)")
    print(f"👎 Negative: {neg} ({neg / total * 100:.1f}%)")
    print(f"😐 Neutral: {neu} ({neu / total * 100:.1f}%)")

    print("\n🔖 Custom Label Distribution:")
    print(result_df['custom_label'].value_counts())

    print("\n🔗 Sentiment vs Custom Label:")
    print(result_df.groupby(['sentiment', 'custom_label']).size().unstack(fill_value=0))

    return result_df


if __name__ == "__main__":
    analyze_replies_with_context()