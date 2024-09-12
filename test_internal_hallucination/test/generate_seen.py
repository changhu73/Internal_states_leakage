import json

MIN_LENGTH = 120
MAX_LENGTH = 135

with open('dataset/cbs_old_news.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

seen_data = []
buffer = ""
i = 0

for line in lines:
    buffer += " " + line
    buffer = buffer.strip()
    
    while len(buffer) >= MIN_LENGTH and len(seen_data) < 5000:
        if len(buffer) > MAX_LENGTH:
            content = buffer[:MAX_LENGTH]
            buffer = buffer[MAX_LENGTH:].strip()
        else:
            content = buffer
            buffer = ""

        title = f"Article {len(seen_data) + 1}"
        seen_data.append({
            "id": len(seen_data) + 1,
            "title": title,
            "content": content
        })

    if len(seen_data) >= 5000:
        break

if len(buffer) >= MIN_LENGTH and len(seen_data) < 5000:
    title = f"Article {len(seen_data) + 1}"
    seen_data.append({
        "id": len(seen_data) + 1,
        "title": title,
        "content": buffer
    })

with open('dataset/seen_news.json', 'w', encoding='utf-8') as f:
    json.dump(seen_data, f, ensure_ascii=False, indent=4)

print("新闻片段已保存到 seen_news.json")
