import pandas as pd
import json

file_path = '/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/Result.csv'
df = pd.read_csv(file_path)

input_data = df.iloc[:, 4]
reference_data = df.iloc[:, 6]

output_data = []
for input_value, reference_value in zip(input_data, reference_data):
    if pd.notna(reference_value):
        output_data.append({
            'input': input_value if pd.notna(input_value) else None,
            'reference': reference_value
        })

for item in output_data:
    input_value = item['input']
    reference_value = item['reference']
    
    if input_value and reference_value and reference_value.startswith(input_value):
        item['reference'] = reference_value[len(input_value):].strip()

output_json_file_path = '/home/guangwei/LLM-COPYRIGHT/copyright_newVersion/data/data.extra.json'
with open(output_json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"数据已处理并保存到 {output_json_file_path}")
