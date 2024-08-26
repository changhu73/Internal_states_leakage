import json

with open('dataset/cbs_old_news.txt', 'r', encoding='utf-8') as f:
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

with open('dataset/seen_news.json', 'w', encoding='utf-8') as f:
    json.dump(seen_data, f, ensure_ascii=False, indent=4)

print("新闻片段已保存到 seen_news.json")
