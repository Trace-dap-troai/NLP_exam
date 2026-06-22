# --------------------------------------------------
# BMW AI CEO Agent
# ETL Pipeline (Senior Architecture):
# 1. Collect live data from RSS feeds with Exception Shielding
# 2. Clean HTML tags & generate Deterministic MD5 IDs
# 3. DistilBERT Sentiment Analysis (Native Token Truncation)
# 4. Incremental Concat to preserve Historical Database
# --------------------------------------------------
import os
import re
import time
import hashlib
import feedparser
import pandas as pd
from datetime import datetime
from transformers import pipeline

# avoid Segmentation Fault  Mac M1
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_FILE = "bmw_live_data.csv"

RSS_SOURCES = {
    "BMW Official Pressroom": "https://www.bmwblog.com/feed/",
    "Google News - BMW Corporate": "https://news.google.com/rss/search?q=BMW+Group+corporate+news+business&hl=en-US&gl=US&ceid=US:en",
    "Automotive News - Competitors": "https://news.google.com/rss/search?q=Mercedes+OR+Audi+OR+Tesla+automotive+strategy+market+share&hl=en-US&gl=US&ceid=US:en",
    "Market Intelligence - EV Trends": "https://news.google.com/rss/search?q=electric+vehicle+market+trends+supply+chain&hl=en-US&gl=US&ceid=US:en",
    "Reddit BMW": "https://www.reddit.com/r/BMW/.rss"
}

def load_sentiment_model():
    print("Loading DistilBERT sentiment model...")
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

def clean_html_text(raw_html):
   
    if not raw_html:
        return ""
    clean_text = re.sub(r'<.*?>', ' ', raw_html)
    clean_text = re.sub(r'&[a-zA-Z0-9]+;', ' ', clean_text)
    return ' '.join(clean_text.split())

def generate_deterministic_id(url):
    """Create unique ID (MD5)"""
    return "bmw_" + hashlib.md5(url.encode('utf-8')).hexdigest()[:16]

def analyze_sentiment(model, text):
    """HuggingFace standazation Token Truncation"""
    try:
        result = model(text, truncation=True, max_length=512)[0]
        return result["label"], round(result["score"], 4)
    except Exception as e:
        print(f" [!] Sentiment parsing error: {e}")
        return "NEUTRAL", 0.50

def collect_articles():
    sentiment_model = load_sentiment_model()
    pipeline_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []


    for source_name, url in RSS_SOURCES.items():
        print(f"Collecting from: {source_name}...")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = getattr(entry, "title", "").strip()
                raw_summary = getattr(entry, "summary", "")
                article_url = getattr(entry, "link", "N/A")
                published = getattr(entry, "published", time.ctime())

                if not title or article_url == "N/A":
                    continue

                clean_summary = clean_html_text(raw_summary)
                content = f"{title}. {clean_summary}"

                label, score = analyze_sentiment(sentiment_model, content)
                doc_id = generate_deterministic_id(article_url)

                new_rows.append({
                    "id": doc_id,
                    "title": title,
                    "source": source_name,
                    "content": content,
                    "link": article_url,
                    "published_date": published,
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "pipeline_run_time": pipeline_timestamp
                })
        except Exception as e:
            # fault per-feed
            print(f" [!!!!] : CAN'T Scrap feed {source_name}. fault: {e}")
    raw_df = pd.DataFrame(new_rows)
    if not raw_df.empty:
        # clean duplicate
        raw_df.drop_duplicates(subset=["link"], keep="first", inplace=True)

    return raw_df


def merge_with_historical_data(new_df):
    """Gộp dồn Incremental giữ nguyên lịch sử DB"""
    if os.path.exists(OUTPUT_FILE):
        print("Found existing historical database. Merging incrementally...")
        old_df = pd.read_csv(OUTPUT_FILE)
        # new + old
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        #clean duplicate keep latest rss news
        combined_df.drop_duplicates(subset=["id"], keep="last", inplace=True)
        return combined_df
    return new_df

def main():
    print("=" * 55)
    print(" BMW Strategic AI Agent - Advanced ETL Pipeline")
    print("=" * 55)
    
    new_df = collect_articles()
    print(f"\nScraped {len(new_df)} live items from RSS feeds.")
    
    final_df = merge_with_historical_data(new_df)
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "=" * 55)
    print(f"PIPELINE FINISHED SUCCESSFULLY!")
    print(f"Total Knowledge Documents in Repository: {len(final_df)}")
    print(f"Database path: ./{OUTPUT_FILE}")
    print("=" * 55)

if __name__ == "__main__":
    main()