import pandas as pd
import json


# seen_data
with open('dataset/seen_news.json', 'r', encoding='utf-8') as f:
    seen_data = json.load(f)


# unseen_data
with open('dataset/unseen_news.json', 'r', encoding='utf-8') as f:
    unseen_data = json.load(f)


# 保存到CSV文件
seen_df = pd.DataFrame(seen_data)
unseen_df = pd.DataFrame(unseen_data)

seen_df.to_csv("dataset/seen_news.csv", index=False, encoding='utf-8')
unseen_df.to_csv("dataset/unseen_news.csv", index=False, encoding='utf-8')
