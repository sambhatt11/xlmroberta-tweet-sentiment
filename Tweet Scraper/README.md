📌 Tweet & Replies Scraper
This is a simple Python script that uses Selenium to scrape the text, date, and language of a specific tweet and all its available replies. It is designed for educational or research purposes only.

🚀 Features
	•	Scrape any single tweet by URL
	•	Extract tweet text, posting date, and language
	•	Automatically scroll to load and collect all visible replies
	•	Save the main tweet and replies to CSV files
	•	Option to run headless or with a visible browser for manual CAPTCHA solving

📂 Output
	•	One CSV file for the main tweet: <tweet_id>_tweet.csv
	•	One CSV file for the replies: <tweet_id>_replies.csv 
Each CSV includes:
	•	Tweet/reply text
	•	Username (for replies)
	•	Date
	•	Language
	•	Tweet ID (for linking replies to the main tweet)

⚙️ Requirements
	•	Python 3.8+
	•	Google Chrome installed
	•	pip install selenium webdriver-manager pandas  

📌 Usage
1. Clone the repo git clone https://github.com/RocketmanLXVII/Tweet-Scraper 
     cd Tweet-Scraper	
2. Run the script python scrape_tweet.py --tweet_url "<FULL_TWEET_URL>"
 Example: python scrape_tweet.py --tweet_url "https://twitter.com/username/status/1234567890"

          A browser will open.
	◦	Solve any CAPTCHA, log in, or accept cookies manually.
	◦	When done, press Enter in the terminal to continue scraping.

⚠️ Notes
	•	Twitter often changes its structure and blocks bots. This script works best if you manually solve login and cookie banners.
	•	This scraper does not bypass advanced bot detection or rate limits.
	•	Use responsibly. This is for educational purposes only — respect Twitter’s Terms of Service.

🗂️ Project Structure
├── scrape_tweet.py
├── README.md
├── <tweet_id>_tweet.csv
├── <tweet_id>_replies.csv

✅ To Do / Improvements
	•	Add automatic login or cookie management
	•	Add sentiment or keyword analysis
	•	Improve language detection accuracy
	•	Containerize with Docker for easier deployment

📄 License
This project is licensed for educational use only. Always comply with the terms and privacy policies of the sites you scrape.

Happy scraping! 🐍✨
