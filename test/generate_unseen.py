import json

with open('dataset/cbs_2024_news.txt', 'r', encoding='utf-8') as f:
    lines = [f.readline().strip() for _ in range(5000)]

seen_data = []

for i, line in enumerate(lines):
    if line:
        title = f"Article {i + 1}"
        seen_data.append({
            "id": i + 1,
            "title": title,
            "content": line
        })

with open('dataset/unseen_news.json', 'w', encoding='utf-8') as f:
    json.dump(seen_data, f, ensure_ascii=False, indent=4)

print("新闻片段已保存到 unseen_news.json")
