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


# Transliterate Romanized Bengali to Unicode
def transliterate_bengali_to_unicode(text):
    bengali_map = {
        'a': 'অ', 'aa': 'আ', 'i': 'ই', 'ii': 'ঈ', 'u': 'উ', 'uu': 'ঊ',
        'e': 'এ', 'o': 'ও',
        'k': 'ক', 'kh': 'খ', 'g': 'গ', 'gh': 'ঘ', 'ng': 'ঙ',
        'ch': 'চ', 't': 'ট', 'th': 'ঠ', 'd': 'ড', 'dh': 'ঢ', 'n': 'ণ',
        'p': 'প', 'ph': 'ফ', 'b': 'ব', 'bh': 'ভ', 'm': 'ম',
        'l': 'ল', 'r': 'র', 's': 'স', 'sh': 'শ', 'h': 'হ',
        'j': 'জ', 'z': 'য', 'y': 'য', 'w': 'ও', 'x': 'ক্স', 'v': 'ভ'
    }

    for key in sorted(bengali_map, key=len, reverse=True):
        text = text.replace(key, bengali_map[key])

    return text.strip()


# Transliterate Romanized Hindi (Hinglish) to Devanagari
def transliterate_hindi_to_devanagari(text):
    hindi_map = {
        'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ee': 'ई', 'u': 'उ', 'oo': 'ऊ',
        'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
        'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ch': 'च', 'chh': 'छ',
        'j': 'ज', 'jh': 'झ', 'tt': 'ट', 'tth': 'ठ', 'dd': 'ड', 'ddh': 'ढ', 'nn': 'ण',
        't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
        'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
        'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व', 'sh': 'श', 's': 'स',
        'h': 'ह', 'z': 'ज़', 'rr': 'ऋ', 'm': 'ं', 'ah': 'ः', 'an': 'ा', 'ka': 'का'
    }

    for key in sorted(hindi_map, key=len, reverse=True):
        text = text.replace(key, hindi_map[key])

    return text.strip()


# Get coarse sentiment
def get_sentiment(model, tokenizer, text):
    if not text.strip():
        return "neutral", 0.0

    inputs = tokenizer(text[:512], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    labels = ['negative', 'neutral', 'positive']
    return labels[probs.index(max(probs))], max(probs)


# Classify fine-grained sentiment with better prompt
def classify_fine_grained_sentiment(zero_shot_pipe, tweet_text, reply_text):
    combined_prompt = f"The tweet says: '{tweet_text}'\nThe reply says: '{reply_text}'\nClassify this reply into one of the following categories."

    fine_labels = {
        "negative": ["insult", "hate", "criticism"],
        "positive": ["praise", "support", "celebration"],
        "neutral": ["inquiry", "statement", "off-topic"]
    }

    base_result = zero_shot_pipe(reply_text, ["negative", "positive", "neutral"])
    base_sentiment = base_result['labels'][0]

    fine_label = zero_shot_pipe(reply_text, fine_labels[base_sentiment])['labels'][0]
    
    return base_sentiment, fine_label


# Get similarity between tweet and reply
def get_similarity(sim_model, tweet_text, reply_text):
    if not tweet_text or not reply_text:
        return 0.0
    embeddings = sim_model.encode([tweet_text, reply_text])
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(cosine_sim, 2)


# Load tweets and replies safely
def load_tweets_and_replies(tweets_file="tweets.csv", replies_file="replies.csv"):
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

    # Rename columns if needed
    if len(df_tweets.columns) >= 2:
        df_tweets.columns = ['tweet_id', 'tweet_text'] + list(df_tweets.columns[2:])
    if len(df_replies.columns) >= 5:
        df_replies.columns = ['reply_text', 'reply_username', 'reply_date', 'language', 'tweet_id'] + list(df_replies.columns[5:])

    # Convert tweet_id to int and fix scientific notation
    df_tweets['tweet_id'] = pd.to_numeric(df_tweets['tweet_id'], errors='coerce').astype('Int64')
    df_replies['tweet_id'] = pd.to_numeric(df_replies['tweet_id'], errors='coerce').astype('Int64')

    return df_tweets, df_replies


# Main function
def analyze_replies_with_context(tweet_csv_path="tweets.csv",
                               replies_csv_path="replies.csv",
                               output_file="replies_with_analysis.csv"):
    df_tweets, df_replies = load_tweets_and_replies(tweet_csv_path, replies_csv_path)

    if df_tweets is None or df_replies is None:
        print("❌ Failed to load datasets.")
        return

    print("🧠 Loading models...")

    # Sentiment model
    sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)
    sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    # Semantic similarity model (multilingual support)
    sim_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

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
            # Improve analysis by transliterating Romanized text
            reply_lang = detect_language(cleaned)
            reply_text_for_analysis = cleaned

            if reply_lang == 'bn' and all(ord(c) < 128 for c in cleaned):  # Romanized Bengali
                reply_text_for_analysis = transliterate_bengali_to_unicode(cleaned)
            elif reply_lang == 'hi' and all(ord(c) < 128 for c in cleaned):  # Hinglish
                reply_text_for_analysis = transliterate_hindi_to_devanagari(cleaned)

            # Get coarse sentiment with lower threshold
            coarse_sentiment, confidence = get_sentiment(sent_model, sent_tokenizer, reply_text_for_analysis)

            # Lower threshold from 0.6 → 0.5
            if confidence < 0.30:
                coarse_sentiment = "neutral"

            # Get fine sentiment strictly within coarse category
            _, fine_sentiment = classify_fine_grained_sentiment(zero_shot, tweet_text, reply_text_for_analysis)

            # Enforce label mapping
            if coarse_sentiment == 'negative' and fine_sentiment in ["praise", "support"]:
                fine_sentiment = "criticism"
            elif coarse_sentiment == 'positive' and fine_sentiment in ["insult", "hate"]:
                fine_sentiment = "support"

            # Get similarity
            similarity_score = get_similarity(sim_model, tweet_text, reply_text_for_analysis)

            # Dynamic threshold based on language and length
            base_threshold = 0.4
            if reply_lang in ['bn', 'hi'] or len(reply_text_for_analysis) > 20:
                base_threshold = 0.45
            relevance_label = "related" if similarity_score > base_threshold else "unrelated"

            results.append({
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "analyzed_text": reply_text_for_analysis,
                "tweet_text": tweet_text,
                "reply_username": getattr(row, 'reply_username', ''),
                "reply_date": getattr(row, 'reply_date', ''),
                "tweet_id": tweet_id,
                "language": reply_lang,
                "coarse_sentiment": coarse_sentiment,
                "fine_sentiment": fine_sentiment,
                "confidence": round(confidence, 2),
                "relevance_score": similarity_score,
                "relevance_label": relevance_label
            })

            print(f"[{idx}] 🧠 {coarse_sentiment} → {fine_sentiment} | {cleaned[:50]}... | 🔍 {relevance_label}")

        except Exception as e:
            continue

    if not results:
        print("❌ No valid replies were processed.")
        return

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved {len(result_df)} replies with analysis to '{output_file}'")

    total = len(result_df)
    pos = len(result_df[result_df['coarse_sentiment'] == 'positive'])
    neg = len(result_df[result_df['coarse_sentiment'] == 'negative'])
    neu = total - pos - neg

    print("\n📊 Public Reply Sentiment Breakdown")
    print(f"👍 Positive: {pos} ({pos / total * 100:.1f}%)")
    print(f"👎 Negative: {neg} ({neg / total * 100:.1f}%)")
    print(f"😐 Neutral: {neu} ({neu / total * 100:.1f}%)")

    print("\n🧾 Sample Results:")
    print(result_df[['cleaned_text', 'coarse_sentiment', 'fine_sentiment']].head())

    return result_df


if __name__ == "__main__":
    analyze_replies_with_context()