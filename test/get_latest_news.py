import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

base_url = 'https://www.cbsnews.com/latest/'
page = 1
news_contents = []

with tqdm(total=500, desc="正在爬取新闻", unit="篇") as pbar:
    while len(news_contents) < 500:
        url = f'{base_url}?page={page}'
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            articles = soup.find_all('article')

            if not articles:
                break

            for article in articles:
                headline = article.find('h4').get_text(strip=True)
                link = article.find('a')['href']
                if "2024" in headline:
                    news_response = requests.get(link)
                    if news_response.status_code == 200:
                        news_soup = BeautifulSoup(news_response.content, 'lxml')
                        paragraphs = news_soup.find_all('p')
                        article_content = "\n".join([p.get_text(strip=True) for p in paragraphs])
                        news_contents.append(f"Title: {headline}\n\n{article_content}\n{'='*80}\n")
                        
                        pbar.update(1)

                        if len(news_contents) >= 500:
                            break
            page += 1
            time.sleep(2)
        else:
            print(f"请求失败，状态码: {response.status_code}")
            break

with open('dataset/cbs_2024_news.txt', 'w', encoding='utf-8') as f:
    for content in news_contents:
        f.write(content)

print("新闻已保存到 cbs_2024_news.txt")
