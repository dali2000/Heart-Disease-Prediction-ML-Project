import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def scrape_health_news(url="https://www.medicalnewstoday.com/categories/heart-disease"):
    """
    Scrapes headlines and links from Medical News Today - Heart Disease section.
    Objective: Extract medical news for context and trend analysis.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"Fetching health news from {url}...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    news_items = []

    # MedicalNewsToday article cards
    cards = soup.find_all("li", class_="css-1y6a00t")

    for card in cards:
        title_tag = card.find("a")
        time_tag = card.find("time")

        if title_tag and title_tag.text.strip():
            headline = title_tag.text.strip()
            link = title_tag.get("href", "")
            timestamp = time_tag.text.strip() if time_tag else datetime.now().strftime("%Y-%m-%d")

            news_items.append({
                "headline": headline,
                "timestamp": timestamp,
                "link": link if link.startswith("http") else f"https://www.medicalnewstoday.com{link}"
            })

    # Fallback: try generic article links
    if not news_items:
        for a in soup.find_all("a", href=True):
            text = a.text.strip()
            href = a["href"]
            if len(text) > 40 and "heart" in text.lower():
                news_items.append({
                    "headline": text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d"),
                    "link": href if href.startswith("http") else f"https://www.medicalnewstoday.com{href}"
                })
            if len(news_items) >= 20:
                break

    return pd.DataFrame(news_items)

if __name__ == "__main__":
    df = scrape_health_news()

    if df is not None and not df.empty:
        print(f"Successfully scraped {len(df)} news items.")
        print(df.head())
        df.to_csv("data/health_news.csv", index=False)
        print("Data saved to data/health_news.csv")
    else:
        print("No news items found.")
