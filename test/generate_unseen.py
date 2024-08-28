import json

MIN_LENGTH = 120
MAX_LENGTH = 135

with open('dataset/cbs_2024_news.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

unseen_data = []
buffer = ""
i = 0

for line in lines:
    buffer += " " + line
    buffer = buffer.strip()
    
    while len(buffer) >= MIN_LENGTH and len(unseen_data) < 5000:
        if len(buffer) > MAX_LENGTH:
            content = buffer[:MAX_LENGTH]
            buffer = buffer[MAX_LENGTH:].strip()
        else:
            content = buffer
            buffer = ""

        title = f"Article {len(unseen_data) + 1}"
        unseen_data.append({
            "id": len(unseen_data) + 1,
            "title": title,
            "content": content
        })

    if len(unseen_data) >= 5000:
        break

if len(buffer) >= MIN_LENGTH and len(unseen_data) < 5000:
    title = f"Article {len(unseen_data) + 1}"
    unseen_data.append({
        "id": len(unseen_data) + 1,
        "title": title,
        "content": buffer
    })

with open('dataset/unseen_news.json', 'w', encoding='utf-8') as f:
    json.dump(unseen_data, f, ensure_ascii=False, indent=4)

print("新闻片段已保存到 unseen_news.json")
