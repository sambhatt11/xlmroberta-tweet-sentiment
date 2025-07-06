# Sentiment-Analyzer
🧠 Tweet Sentiment Classifier using XLM-RoBERTa
This project performs sentiment analysis on English and Bengali tweets scraped using snscrape, with classification powered by the XLM-RoBERTa transformer model.

Tweets are categorized into primary sentiment labels — Positive, Negative, and Neutral — and further subclassified into fine-grained emotions such as:

💬 Praise

🛡️ Supportive

⚠️ Criticism

🚫 Insult

📌 Key Features:
📥 scraper.py script for collecting tweets using snscrape

🌐 Handles both English and Bengali (native & Romanized)

🤖 Uses xlm-roberta-base fine-tuned for multilingual sentiment classification

🏷️ Two-level classification: Sentiment + Subsentiment

📊 Detailed evaluation with accuracy, F1-score, and confusion matrix

This project demonstrates how transformer-based models can be applied to multilingual, real-world social media data, offering both coarse and fine sentiment understanding.

