from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import argparse


def setup_browser(headless=False):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")  # Disable extensions causing issues
    chrome_options.add_argument("--lang=en")  # Set default language

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


# Clean tweet text
def clean_tweet_text(text):
    return text.replace('\n', ' ').strip()


# Detect tweet language
def detect_tweet_language(tweet_element):
    try:
        return tweet_element.find_element(By.XPATH, './/div[@data-testid="tweetText"]').get_attribute("lang")
    except:
        return "unknown"


# Scrape single tweet
def scrape_single_tweet(driver, tweet_url):
    print(f"🌐 Visiting tweet: {tweet_url}")
    driver.get(tweet_url)

    print("⏳ Please solve CAPTCHA or accept cookies manually...")
    input("➡️ Press Enter after solving any warnings or logging in...")

    print("⏳ Waiting for tweet to load...")
    time.sleep(10)

    try:
        tweet_element = driver.find_element(By.XPATH, '//article[@data-testid="tweet"]')
        tweet_text_element = tweet_element.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
        tweet_text = clean_tweet_text(tweet_text_element.text)
        tweet_date_element = tweet_element.find_element(By.XPATH, './/time')
        tweet_date = tweet_date_element.get_attribute("datetime")
        tweet_lang = detect_tweet_language(tweet_element)

        tweet_data = {
            "tweet_id": hash(tweet_text),
            "tweet_text": tweet_text,
            "tweet_date": tweet_date,
            "language": tweet_lang
        }

        print("✅ Successfully scraped tweet:")
        print(f"📝 {tweet_text[:100]}...")

        return tweet_data

    except Exception as e:
        print("❌ Could not find tweet:", str(e))
        return None


# Scrape all available replies
def scrape_replies(driver, main_tweet_text, max_scrolls=20, delay=5):
    print("💬 Loading replies...")
    reply_data = []

    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0

    while scroll_count < max_scrolls:
        print(f"🔄 Scroll {scroll_count + 1}/{max_scrolls} to load more replies...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

        try:
            reply_elements = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
            for reply in reply_elements:
                try:
                    # Try to get user handle
                    user_span = reply.find_elements(By.XPATH, './/span[starts-with(text(), "@")]')
                    reply_username = user_span[0].text if user_span else "unknown_user"

                    # Get reply text
                    reply_text_element = reply.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
                    reply_text = clean_tweet_text(reply_text_element.text)

                    # Skip empty or main tweet itself
                    if not reply_text or reply_text == main_tweet_text:
                        continue

                    # Get date
                    reply_date_element = reply.find_element(By.XPATH, './/time')
                    reply_date = reply_date_element.get_attribute("datetime")

                    # Get language
                    reply_lang = detect_tweet_language(reply)

                    # Save reply
                    reply_data.append({
                        "reply_text": reply_text,
                        "reply_username": reply_username,
                        "reply_date": reply_date,
                        "language": reply_lang
                    })

                except Exception as e:
                    continue  # Skip problematic elements

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("🔚 No more replies to load.")
                break
            last_height = new_height
            scroll_count += 1

        except Exception as e:
            print("⚠️ Error fetching replies:", str(e))
            break

    return reply_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape one tweet and all replies from a specific tweet URL")
    parser.add_argument("--tweet_url", "-u", required=True, help="URL of the tweet to scrape")
    args = parser.parse_args()

    print("🚀 Starting browser...")
    driver = setup_browser(headless=False)  # Run in visible mode to bypass privacy blocker

    try:
        # Step 1: Scrape the main tweet
        tweet_info = scrape_single_tweet(driver, args.tweet_url)

        if not tweet_info:
            print("❌ Failed to scrape tweet. Exiting.")
            exit()

        # Step 2: Scrape replies
        print("💬 Getting replies for this tweet...")
        replies = scrape_replies(driver, main_tweet_text=tweet_info["tweet_text"])

        if replies:
            # Add tweet_id to all replies
            for reply in replies:
                reply["tweet_id"] = tweet_info["tweet_id"]

            replies_df = pd.DataFrame(replies)
            replies_file = f"{tweet_info['tweet_id']}_replies.csv"
            replies_df.to_csv(replies_file, index=False, encoding='utf-8-sig')
            print(f"✅ Saved {len(replies)} replies to '{replies_file}'")
        else:
            print("ℹ️ No replies were found.")

        # Save main tweet
        tweet_df = pd.DataFrame([tweet_info])
        tweet_file = f"{tweet_info['tweet_id']}_tweet.csv"
        tweet_df.to_csv(tweet_file, index=False, encoding='utf-8-sig')
        print(f"✅ Saved tweet to '{tweet_file}'")

    finally:
        input("\n👋 Press Enter to close the browser...")
        driver.quit()